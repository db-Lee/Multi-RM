import argparse
import json
import os
from tqdm import tqdm

import torch
import multiprocessing as mp
from datasets import load_dataset

from discriminative.utils import split_dataset_for_gpus
from discriminative.reward_model import RewardModel

def flatten_all_data(dataset):
    flattened = []
    
    for q_idx, data in enumerate(dataset):
        for cot_idx, cot in enumerate(data['cots']):
            steps = [step.strip().replace(' \n\n\n\n', '') for step in cot]
            question = data['question'].strip().replace(' \n\n\n\n', '')
            
            updated_steps = [f'{step} \n\n\n\n' for step in steps]
            steps_all = f'{question} \n\n' + ''.join(updated_steps)
            
            flattened.append({
                'steps_all': steps_all,
                'q_idx': q_idx,
                'cot_idx': cot_idx
            })
    
    return flattened

def reconstruct_results(dataset, reward_results):
    results = []
    for data in dataset:
        result_data = {
            'q_id': data['q_id'],
            'cot_ids': data['cot_ids'],
            'rewards': [None] * len(data['cots'])
        }
        results.append(result_data)
    
    for reward_item in reward_results:
        q_idx = reward_item['q_idx']
        cot_idx = reward_item['cot_idx']
        reward = reward_item['reward']
        results[q_idx]['rewards'][cot_idx] = reward
    
    return results

def process_gpu_batch(gpu_id, dataset, args, temp_file=None):
    print(f"Process {gpu_id}: Initializing PRM on GPU {gpu_id}...")
    
    reward_model = RewardModel(model_id=args.model_id, aggregation="full", device=torch.device(f'cuda:{gpu_id}'))
    
    flattened_data = flatten_all_data(dataset)
    print(f"Process {gpu_id}: PRM initialized. Starting processing...")
    print(f"Process {gpu_id}: Total CoTs to process: {len(flattened_data)}")
    
    reward_results = []
    for i in tqdm(
        range(0, len(flattened_data), args.per_device_batch_size), 
        desc=f'Process {gpu_id}: Processing CoT batches'
    ):
        batch = flattened_data[i:i + args.per_device_batch_size]
        batch_steps = [item['steps_all'] for item in batch]
        batch_rewards = reward_model(batch_steps)
        
        for reward, item in zip(batch_rewards, batch):
            reward_results.append({
                'reward': reward,
                'q_idx': item['q_idx'],
                'cot_idx': item['cot_idx']
            })
    
    print(f"Process {gpu_id}: Reconstructing results...")
    output_results = reconstruct_results(dataset, reward_results)
    
    if temp_file is not None:
        with open(temp_file, "w") as f:
            json.dump(output_results, f, indent=4)
    else:
        return output_results

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Process rewards for discriminative reward models')
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--data_path', type=str, default="dongboklee/test")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument("--category", type=str, default="all", choices={
        'law', 'psychology', 'chemistry', 'biology', 'physics', 
        'history', 'economics', 'math', 'business', 'philosophy', 
        'health', 'engineering', 'computer_science', 'other', "all", 'gsm8k', 'math'
    }, help="Category of problems to process")
    parser.add_argument('--per_device_batch_size', type=int, default=8, help="Batch size for CoT processing")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs, creating {num_gpus} processes (1 GPU per process)")
    
    if num_gpus > 1:
        mp.set_start_method('spawn', force=True)

    if args.category == "all":
        category_list = ['law', 'psychology', 'chemistry', 'biology', 'physics', 
                        'history', 'economics', 'math', 'business', 'philosophy', 
                        'health', 'engineering', 'computer_science', 'other']
    else:
        category_list = [args.category]

    for category in category_list:
        print(f"Loading dataset for category: {category}...")
        
        try:
            dataset = load_dataset(args.data_path, split=category)
        except:
            with open(os.path.join(args.input_dir, f"{category}.json"), "r") as f:            
                dataset = json.load(f)        
        
        print(f"Dataset loaded: {len(dataset)} items")
        
        print(f"Using {num_gpus} processes for processing")
        dataset_batches = split_dataset_for_gpus(dataset, num_gpus)
        
        processes, temp_file_list = [], []
        for gpu_id in range(num_gpus):
            if len(dataset_batches[gpu_id]) > 0:
                temp_file = os.path.join(args.output_dir, f"{category}_temp_file_{gpu_id}.json")
                temp_file_list.append(temp_file)
                
                p = mp.Process(
                    target=process_gpu_batch,
                    args=(gpu_id, dataset_batches[gpu_id], args, temp_file)
                )
                processes.append(p)
                p.start()
                print(f"Started process on GPU {gpu_id}")
        
        for p in processes:
            p.join()
        
        output_results = []
        for temp_file in temp_file_list:
            with open(temp_file, "r") as f:
                output_results.extend(json.load(f))
            os.remove(temp_file)

        output_file = os.path.join(args.output_dir, f"{category}_reward.json")
        with open(output_file, "w") as f:
            json.dump(output_results, f, indent=4)
            
        print(f"Results for {category} saved to {output_file}")
        print(f"Processed {len(output_results)} items using {num_gpus} processes with 1 GPU each")
        
if __name__ == '__main__':
    main()