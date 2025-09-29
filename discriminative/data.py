import os
import json
from tqdm import tqdm
from copy import deepcopy

from torch.utils.data import Dataset
from datasets import load_dataset

def merge_dicts(dict_list):
    merged = deepcopy(dict_list[0])
    for d in dict_list[1:]:
        for k, v in d.items():
            merged[k].extend(v)
    return merged

def get_category_dataset(data_path, category):
    category_list = ['law', 'psychology', 'chemistry', 'biology', 
                     'physics', 'history', 'economics', 'math', 'business', 
                     'philosophy', 'health', 'engineering', 'computer_science', 'other']
    if category == "prm800k":
        category_list = ["prm800k"]
    elif category == "all":
        pass
    elif category in category_list:
        category_list = [category]
    else:
        raise NotImplementedError
    
    combined_dataset = []
    for category in category_list:
        try:
            dataset = load_dataset(data_path, split=category)
            dataset = [ d for d in dataset ]
        except:
            with open(os.path.join(data_path, f"{category}.json"), "r") as f:
                dataset = json.load(f)        
        combined_dataset.extend(dataset)
    return combined_dataset

def tokenize_step(step, label, tokenizer, mask_id=-100, label_last_n=None):
    tokenized = tokenizer(step, add_special_tokens=False)
    step_len = len(tokenized.input_ids)
    
    if label_last_n is None or label_last_n >= step_len:
        labels = [label] * step_len
    else:
        labels = [mask_id] * (step_len - label_last_n) + [label] * label_last_n
    
    tokenized['labels'] = labels
    return tokenized

def tokenize_one_data(data, tokenizer, mask_id=-100, label_last_n=None, max_length=None, orm=False, 
                     labels=None):
    """Tokenize one data point with given labels."""
    # Question
    question_tok = tokenizer(f"{data['question']} \n\n")
    question_tok['labels'] = [mask_id] * len(question_tok.input_ids)
    
    if labels is None:
        return []
    
    steps_tok = []
    
    # Process steps
    for i, step in enumerate(data['cot']):
        if orm:
            # ORM: only predict on last step
            if (data["parsed_answer"] is None) or (data["answer"] is None):
                label = 0 if -1 in labels else 1
            else:
                label = 1 if data["parsed_answer"] == data["answer"] else 0
            step_label_last_n = label_last_n if i == len(data['cot']) - 1 else 0
        else:
            # PRM: predict on all steps
            label = 0 if labels[i] == -1 else 1
            step_label_last_n = label_last_n
            
        step_tok = tokenize_step(f'{step} \n\n\n\n', label, tokenizer, mask_id, step_label_last_n)
        steps_tok.append(step_tok)
        
        # PRM: stop at first incorrect step
        if not orm and label == 0:
            break
    
    tokenized = merge_dicts([question_tok] + steps_tok)
    return tokenized if max_length is None or len(tokenized.input_ids) <= max_length else None

def tokenize_dataset(data_path, category, tokenizer, mask_id=-100, label_last_n=None, 
                    max_length=None, task_type="dORM"):
    
    dataset = get_category_dataset(data_path, category)    
    orm = task_type == "dORM"
    
    total_tokenized = []
    for data in tqdm(dataset):
        if 'labels' not in data or \
            data['labels'] is None or \
                len(data['labels']) == 0:
            continue
        
        labels = data['labels']        
        tokenized = tokenize_one_data(data, tokenizer, mask_id, label_last_n, 
                                         max_length, orm, labels)
        if tokenized is not None:
            total_tokenized.append(tokenized)
    
    return total_tokenized

class TokenizedDataset(Dataset):
    def __init__(self, 
        data_path, 
        category,
        tokenizer,
        label_mask_token_id=-100, 
        label_last_n=1,
        max_length=None, 
        task_type="dORM"
    ):
        super().__init__()
        assert task_type in ["dORM", "dPRM"]
        self.tokenized_data = tokenize_dataset(
            data_path=data_path, 
            category=category, 
            tokenizer=tokenizer, 
            mask_id=label_mask_token_id, 
            label_last_n=label_last_n, 
            max_length=max_length, 
            task_type=task_type
        )
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, i):
        return self.tokenized_data[i]