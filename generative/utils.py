import os
import json
from tqdm import tqdm

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
    batch_size = len(dataset) // num_gpus
    
    batches = []
    for i in range(num_gpus):
        start_idx = i * batch_size
        if i == num_gpus - 1:  # Last GPU gets remaining items
            end_idx = len(dataset)
        else:
            end_idx = (i + 1) * batch_size
        
        if isinstance(dataset, Dataset):
            batch_data = dataset.select(range(start_idx, end_idx))
        else:
            batch_data = dataset[start_idx:end_idx]
        
        batches.append(batch_data)
    
    return batches