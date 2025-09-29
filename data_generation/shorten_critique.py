import os
import json
import re
import argparse

def shorten(text, task_type="gORM", label=None):
    """Trims the text based on task type and assigns labels for ORM."""
    
    # Check if </think> token exists
    if "</think>" not in text:
        return text
    
    # Check that there's exactly one </think> token
    think_count = text.count("</think>")
    if think_count != 1:
        return text
    
    # Split on </think>
    parts = text.split("</think>", 1)
    text1 = parts[0]
    remaining = parts[1] if len(parts) > 1 else ""
    
    if task_type == "gORM":
        # For gORM: Delete everything after </think> and assign based on label
        if label == 1:
            verdict = "Yes"
        elif label == -1:
            verdict = "No"
        else:
            raise NotImplementedError
        result = text1 + "</think>\n\nVerification: Is the answer correct (Yes/No)? " + verdict
        return result
    
    elif task_type == "gPRM":
        # For PRM: Delete everything after the pattern
        pattern = r'Is the solution correct\?\s*(Yes|No)'
        match = re.search(pattern, remaining, re.IGNORECASE)
        
        if match:
            # Keep up to and including the pattern, remove everything after
            text2_plus_pattern = remaining[:match.end()]
            result = text1 + "</think>" + text2_plus_pattern
            return result
        else:
            return text
    
    return text

def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument("--model_id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--category", type=str, default="law", 
                       choices=['law', 'psychology', 'chemistry', 'biology', 'physics', 
                               'history', 'economics', 'math', 'business', 'philosophy', 
                               'health', 'engineering', 'computer_science', 'other', 'prm800k', 'all'])
    parser.add_argument("--task_type", type=str, default="orm", choices=['gORM', 'gPRM'])
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
        print("Using same output directory as input directory")
    
    categories = (['law', 'psychology', 'chemistry', 'biology', 'physics', 
                   'history', 'economics', 'math', 'business', 'philosophy', 
                   'health', 'engineering', 'computer_science', 'other'] 
                  if args.category == 'all' else [args.category])
    
    for category in categories:
        input_file = os.path.join(args.input_dir, f"preprocessed_{category}.json")
        output_file = os.path.join(args.output_dir, f"preprocessed_{category}.json")
        
        with open(input_file, "r") as f:
            dataset = json.load(f)
        
        # Process each item in the dataset
        check_input = True
        new_dataset = []
        for data in dataset:
            new_data = data.copy()
            new_data["critique"] = shorten(data["critique"], args.task_type, data["label"])
            if check_input:
                print(new_data["critique"])
                check_input = False
            new_dataset.append(new_data)
        
        # Save the processed dataset
        with open(output_file, "w") as f:
            json.dump(new_dataset, f, indent=2)
        
        print(f"Processed {category} dataset with task_type={args.task_type}")

if __name__ == "__main__":
    main()