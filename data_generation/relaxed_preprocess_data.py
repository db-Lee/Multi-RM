import os
import re
import json
import argparse
import numpy as np
import random
from tqdm import tqdm

from transformers import AutoTokenizer
from data_generation.utils import parse_prm_label, truncate_after_last_boxed_step

def is_valid_label_format(label, K):
    # Check 1: No np.nan
    if any(np.isnan(x) for x in label):
        return False

    # Case 1: all 1s
    if all(x == 1 for x in label):
        return len(label) == K

    # Case 2: ends with exactly one -1, and all before it are 1
    if label and label[-1] == -1:
        if label.count(-1) != 1:
            return False
        if not all(x == 1 for x in label[:-1]):
            return False
        return len(label) <= K

    return False

def get_orm_label(data):
    if (data["parsed_answer"] is None) or (data["answer"] is None):
        return -1 if -1 in data["labels"] else 1
    else:
        return 1 if data["parsed_answer"] == data["answer"] else -1

def get_first_error_step_index(labels):
    for i, label in enumerate(labels):
        if label == -1:
            return i
    return len(labels)

def normalize_process_labels(labels):
    if not labels:
        return []
    
    normalized = labels.copy()
    first_error_pos = get_first_error_step_index(labels)
    
    # Make all labels before first error position become 1
    for i in range(0, first_error_pos):
        normalized[i] = 1
    
    # Make all labels after first error position become -1
    for i in range(first_error_pos, len(normalized)):
        normalized[i] = -1
    
    return normalized

def get_prm_label(data):
    # Use normalized original labels
    full_labels = normalize_process_labels(data["labels"])
    
    # Cut off at the first -1 (original behavior)
    label = []
    for l in full_labels:
        label.append(l)
        if l == -1:
            break    
    
    return label

class DatasetPreprocessor:
    def __init__(self, args):
        self.args = args
        self.processor = {
            "preprocess": truncate_after_last_boxed_step,
            "parse label": parse_prm_label,
            "get label": get_prm_label,
            "parse condition": lambda parsed_label, K: is_valid_label_format(parsed_label, K),
            "correctness condition": lambda parsed_label, label: (label == -1) == (-1 in parsed_label),
            "get yes_or_no": lambda parsed_label: "No" if -1 in parsed_label else "Yes",
            "format": lambda critique, yes_or_no: f"<think>\nLet's verify step by step:{critique}\nIs the solution correct? {yes_or_no}",
            "process cot": lambda cot, label: cot[:label.index(-1)+1] if -1 in label else cot,
            "check positive": lambda label: all([l==1 for l in label]),
            "check negative": lambda label: -1 in label
        }
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    def _process_single_example(self, data, seen_critiques):
        # process critique
        critique = self.processor["preprocess"](data["critique"])
             
        # Check for duplicates
        if critique in seen_critiques:
            return None, "duplicate"
        
        # Check for </think> token
        if not ("</think>" in critique):
            return None, "think"
        
        # Check for Chinese characters
        if bool(re.search(r'[\u4e00-\u9fff]', critique)):
            return None, "chinese"
        
        # Parse and get labels
        parsed_label = self.processor["parse label"](critique)
        label = self.processor["get label"](data)
        
        # 0 cot
        if len(data["cot"]) == 0:
            return None, "0 cot"
        
        # not parsable        
        if not self.processor["parse condition"](parsed_label, len(data["cot"])):
            return None, "not parsable"
        
        # correctness 
        final_answer_correct = get_orm_label(data)       
        if not self.processor["correctness condition"](parsed_label, final_answer_correct):
            return None, "incorrect label"
        
        # Check token length
        if self.args.max_tokens > 0:
            tokenized = self.tokenizer(critique, add_special_tokens=False)
            critique_length = len(tokenized["input_ids"])
            if critique_length > self.args.max_tokens:
                return None, "length"
        
        # Add to seen critiques
        seen_critiques.add(critique)
        
        # postprocess
        formatted_critique = self.processor["format"](critique, self.processor["get yes_or_no"](parsed_label))
                
        return {
            "q_id": data["q_id"],
            "question": data["question"],
            "cot_id": data["cot_id"],             
            "cot": self.processor["process cot"](data["cot"], label),
            "critique": formatted_critique,
            "labels": parsed_label,
        }, None
    
    def balance_examples(self, examples):
        """Balance positive and negative examples by undersampling the majority class"""
            
        # Separate positive and negative examples
        positive_examples = [ex for ex in examples if self.processor["check positive"](ex["labels"])]
        negative_examples = [ex for ex in examples if self.processor["check negative"](ex["labels"])]
        
        n_positive = len(positive_examples)
        n_negative = len(negative_examples)
        
        print(f"\n=== Balancing Examples ===")
        print(f"Before: {n_positive} positive, {n_negative} negative")
        
        if n_positive > n_negative:
            random.seed(self.args.seed)
            positive_examples = random.sample(positive_examples, n_negative)
            print(f"After: {len(positive_examples)} positive, {n_negative} negative (undersampled positive)")
        elif n_negative > n_positive:
            random.seed(self.args.seed)
            negative_examples = random.sample(negative_examples, n_positive)
            print(f"After: {n_positive} positive, {len(negative_examples)} negative (undersampled negative)")
        else:
            print(f"Already balanced: {n_positive} positive, {n_negative} negative")
        
        balanced_examples = positive_examples + negative_examples
        random.seed(self.args.seed)
        random.shuffle(balanced_examples)
        
        return balanced_examples
    
    def preprocess_dataset(self):
        """Main preprocessing function"""
        # Load dataset
        input_path = os.path.join(self.args.input_dir, f"{self.args.category}.json")
        with open(input_path, "r") as f:
            dataset = json.load(f)
            
        # Track statistics
        skip_counts = {
            "duplicate": 0, "think": 0, "chinese": 0, "0 cot": 0,
            "not parsable": 0, "incorrect label": 0, "length": 0
        }
        
        valid_examples = []
        seen_critiques = set()
        
        # Process each example
        for data in tqdm(dataset):
            if 'labels' not in data or \
                data['labels'] is None or \
                    len(data['labels']) == 0:
                continue
            
            processed, skip_reason = self._process_single_example(data, seen_critiques)

            if processed is None:
                skip_counts[skip_reason] += 1
                continue
            
            valid_examples.append(processed)
        
        valid_examples = self.balance_examples(valid_examples)        
        self._print_statistics(len(dataset), len(valid_examples), skip_counts, valid_examples)
        self._save_results(valid_examples)
        
        # Print first example for verification
        if valid_examples:
            print("\n=== Sample Example ===")
            print(valid_examples[0]["critique"])
        
        return valid_examples
    
    def _print_statistics(self, total, valid_count, skip_counts, examples):
        """Print preprocessing statistics"""
        print(f"\nFinal examples after preprocessing: {valid_count} / {total}")
        for skip_type, count in skip_counts.items():
            print(f"# {skip_type} skipped: {count}")
        
        # Label distribution
        pos_count = sum(1 for ex in examples if self.processor["check positive"](ex["labels"]))
        neg_count = len(examples) - pos_count
        
        if examples:
            print(f"\n=== Label Distribution ===")
            print(f"Positive examples (correct): {pos_count} ({pos_count/len(examples):.2%})")
            print(f"Negative examples (incorrect): {neg_count} ({neg_count/len(examples):.2%})")
            print(f"Total examples: {len(examples)}")
            
            # Print (q_id, cot_id) diversity statistics
            unique_pairs = set((ex["q_id"], ex["cot_id"]) for ex in examples)
            print(f"\n=== Diversity Statistics ===")
            print(f"Unique (q_id, cot_id) pairs: {len(unique_pairs)}")
            print(f"Examples per unique pair (avg): {len(examples) / len(unique_pairs):.2f}")
        else:
            print("\n=== Label Distribution ===\nNo examples found")
    
    def _save_results(self, examples):
        """Save preprocessed examples to file"""
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        output_path = os.path.join(
            self.args.output_dir, 
            f"preprocessed_{self.args.category}.json"
        )
        
        with open(output_path, "w") as f:
            json.dump(examples, f, indent=4)
        
        print(f"Results are saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument("--model_id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--category", type=str, default="all", 
                       choices=['law', 'psychology', 'chemistry', 'biology', 'physics', 
                               'history', 'economics', 'math', 'business', 'philosophy', 
                               'health', 'engineering', 'computer_science', 'other', 'prm800k', 'all'])
    parser.add_argument("--max_tokens", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    categories = (['law', 'psychology', 'chemistry', 'biology', 'physics', 
                   'history', 'economics', 'math', 'business', 'philosophy', 
                   'health', 'engineering', 'computer_science', 'other'] 
                  if args.category == 'all' else [args.category])
    
    for category in categories:
        args.category = category
        print(f"Processing category: {category}")
        
        preprocessor = DatasetPreprocessor(args)
        preprocessor.preprocess_dataset()

if __name__ == "__main__":
    main()