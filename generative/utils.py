import os
import json
import argparse
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import Dataset, concatenate_datasets, load_dataset

from generative.data import preprocess_dataset
from generative.prompt_formats import ORM_PROMPT_FORMAT, PRM_PROMPT_FORMAT

def get_dataset(configs, tokenizer):
    task_type, data_path, category = configs.task_type, configs.train_data_path, configs.category
    prompt_format = ORM_PROMPT_FORMAT if task_type == "gORM" else PRM_PROMPT_FORMAT
    def _load_dataset(_category):
        try:
            dataset = load_dataset(data_path, split=_category)
            dataset = [ d for d in dataset ]
        except:
            with open(os.path.join(data_path, f"{_category}.json"), "r") as f:
                dataset = json.load(f)
        
        formatted_dataset = []
        for data in tqdm(dataset, desc=f"Processing {_category}"):
            prompt = prompt_format(_category, data["question"], data["cot"])
            formatted_dataset.append({
                "prompt": f"<｜User｜>{prompt}",
                "completion": f"<｜Assistant｜>{data['critique']}"
            })           
    
        return Dataset.from_list(formatted_dataset)
    
    if category == "all":
        categories = ['law', 'psychology', 'chemistry', 'biology', 'physics', 'history', 
                     'economics', 'math', 'business', 'philosophy', 'health', 'engineering', 
                     'computer_science', 'other']
        dataset = concatenate_datasets([
            _load_dataset(category) for category in categories
        ])
    else:
        dataset = _load_dataset(category)
        
    dataset = preprocess_dataset(dataset, tokenizer)
        
    return dataset

def split_dataset_for_gpus(dataset, num_gpus):
    """Split dataset list into batches for each GPU"""
    batch_size = len(dataset) // num_gpus
    
    batches = []
    for i in range(num_gpus):
        start_idx = i * batch_size
        if i == num_gpus - 1:  # Last GPU gets remaining items
            end_idx = len(dataset)
        else:
            end_idx = (i + 1) * batch_size
        
        batch_data = dataset[start_idx:end_idx]
        batches.append(batch_data)
    
    return batches

def merge_adapter_and_save_temp(input_dir, output_dir):
    adapter_config_path = os.path.join(input_dir, "adapter_config.json")
    
    if os.path.exists(adapter_config_path):
        print("Adapter found, merging with base model...")
        
        temp_dir = os.path.join(output_dir, "tmp")
        if os.path.exists(temp_dir):
            print("Merged model found, end merging.")
            return
    
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        base_model_name = adapter_config.get("base_model_name_or_path")
        if not base_model_name:
            raise ValueError("base_model_name_or_path not found in adapter_config.json")
        
        # Load and merge
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, device_map="cpu", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, input_dir)
        merged_model = model.merge_and_unload()

        # Tokenzier
        tokenizer = AutoTokenizer.from_pretrained(input_dir)    
        
        # Save to temp directory
        merged_model.save_pretrained(temp_dir, safe_serialization=True)    
        tokenizer.save_pretrained(temp_dir)
        print("Model merging completed")
    else:
        print("No adapter found, using checkpoint directly")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir
    merge_adapter_and_save_temp(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()