import os
import gc
import json
import time
import multiprocessing as mp
import argparse

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset

from data_generation.utils import split_dataset_for_gpus
from generative.prompt_formats import ORM_PROMPT_FORMAT, DATA_PRM_PROMPT_FORMAT

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ModelWorker:
    """Worker class that maintains a persistent model instance."""
    
    def __init__(self, process_id, gpu_ids, args):
        # Set GPU visibility
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        
        self.process_id = process_id
        self.args = args
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        
        # Initialize model
        print(f"Process {process_id} (GPUs {gpu_ids}): Initializing vLLM...")
        self.llm = LLM(
            model=args.model_id,
            tokenizer=args.model_id,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=0.95,
            trust_remote_code=True
        )
        print(f"Process {process_id}: Model ready")
        
        # Set up generation parameters
        self.sampling_params = SamplingParams(
            n=args.n_generation,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            stop=None
        )
        
        # Set prompt format
        self.prompt_format = ORM_PROMPT_FORMAT if args.task_type == "gORM" else DATA_PRM_PROMPT_FORMAT
    
    def process_batch(self, category, dataset):
        # Prepare all prompts for the batch
        formatted_prompts = []
        for data in dataset:
            # prompt
            prompt = self.prompt_format(category, data["question"], data["cot"])
            prompt = self.tokenizer.apply_chat_template(
                [{'role': "user", "content": prompt}], 
                tokenize=False, add_generation_prompt=True, add_special_tokens=False
            ) + "Let's verify step by step:"

            # vllm add bos token. So we should remove it.
            bos_token = self.tokenizer.bos_token
            if bos_token is not None and prompt.startswith(bos_token):
                prompt = prompt[len(bos_token):]
            
            formatted_prompts.append(prompt)
            
        # Generate
        outputs = self.llm.generate(formatted_prompts, self.sampling_params, use_tqdm=True)
        
        # Process outputs
        processed_results = []
        for output, item in zip(outputs, dataset):
            for completion in output.outputs:                
                result = item.copy()
                result["critique"] = completion.text
                processed_results.append(result)
        
        return processed_results

def create_cache(args):
    """Initialize vLLM once to create cache, then delete it."""
    print("=" * 80)
    print("Creating vLLM kernel cache...")
    print(f"Model: {args.model_id}")
    print("=" * 80)
    
    # Create temporary LLM instance to build cache
    print("Initializing temporary vLLM instance (this may take a minute)...")
    
    temp_llm = LLM(
        model=args.model_id,
        tokenizer=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.95,
        trust_remote_code=True
    )
        
    print("Cache created successfully. Cleaning up temporary model...")
    
    # Delete the model and free GPU memory
    del temp_llm
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Temporary model deleted. Cache is ready for use.")
    print("=" * 80)
    print()


def worker_process(process_id, gpu_ids, task_queue, result_queue, args):
    """Worker process that processes tasks from queue."""
    # Create worker with persistent model
    worker = ModelWorker(process_id, gpu_ids, args)
    
    # Process tasks from queue
    while True:
        task = task_queue.get()
        if task is None:  # Shutdown signal
            break
        
        category, dataset_chunk = task
        results = worker.process_batch(category, dataset_chunk)
        result_queue.put(results)
    
    print(f"Process {process_id}: Shutting down")


def main():    
    parser = argparse.ArgumentParser(description="Generate step-by-step verification critique data")
    
    # I/O arguments
    parser.add_argument("--input_dir", type=str, default="dongboklee/train")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--category", type=str, default="all", choices={
        'law', 'psychology', 'chemistry', 'biology', 'physics', 
        'history', 'economics', 'math', 'business', 'philosophy', 
        'health', 'engineering', 'computer_science', 'other', 'prm800k', 'all'
    }, help="Category of problems to process")
    parser.add_argument("--task_type", type=str, default="gORM", choices={"gORM", "gPRM"})
    
    # Generation arguments
    parser.add_argument("--model_id", type=str, default="Qwen/QwQ-32B")
    parser.add_argument("--n_generation", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--skip_cache", action="store_true")
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    assert num_gpus % args.tensor_parallel_size == 0, \
        f"GPUs ({num_gpus}) must be divisible by tensor_parallel_size ({args.tensor_parallel_size})"
    
    num_processes = num_gpus // args.tensor_parallel_size
    print(f"Using {num_gpus} GPUs with {num_processes} processes")
    
    # Determine categories to process
    if args.category == "all":
        categories = ['law', 'psychology', 'chemistry', 'biology', 'physics', 
                     'history', 'economics', 'math', 'business', 'philosophy', 
                     'health', 'engineering', 'computer_science', 'other', 'prm800k']
    else:
        categories = [args.category]
    
    print(f"Will process categories: {categories}")
    
    # Create cache if not skipping
    if not args.skip_cache:
        create_cache(args)
        time.sleep(10)
    else:
        print("Skipping cache creation (assuming cache already exists)\n")
    
    print(f"Using {num_processes} processes for parallel processing")
    mp.set_start_method('spawn', force=True)
    
    manager = mp.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    
    # Start worker processes
    processes = []
    for i in range(num_processes):
        gpu_ids = list(range(i * args.tensor_parallel_size, 
                            (i + 1) * args.tensor_parallel_size))
        p = mp.Process(target=worker_process, 
                        args=(i, gpu_ids, task_queue, result_queue, args))
        p.start()
        processes.append(p)
        print(f"Started worker process {i} with GPUs {gpu_ids}")
    
    time.sleep(10)
    
    # Process each category
    for category in categories:
        try:
            dataset = load_dataset(args.data_path, split=category)
            dataset = [ d for d in dataset ]
        except:
            with open(os.path.join(args.input_dir, f"{category}.json"), "r") as f:
                dataset = json.load(f)
                
        print(f"  Loaded {len(dataset)} items")
        
        # Split and distribute work
        chunks = split_dataset_for_gpus(dataset, num_processes)
        active_workers = 0
        
        for i, chunk in enumerate(chunks):
            if len(chunk) > 0:
                task_queue.put((category, chunk))
                active_workers += 1
                print(f"  Assigned {len(chunk)} items to process {i}")
        
        # Collect results
        all_results = []
        for _ in range(active_workers):
            results = result_queue.get()
            all_results.extend(results)
        
        # Save results
        output_file = os.path.join(args.output_dir, f"{category}.json")
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=4)
        
        print(f"  Saved {len(all_results)} results")
        print(f"  Output: {output_file}\n")
    
    # Shutdown workers
    print("Shutting down workers...")
    for _ in range(num_processes):
        task_queue.put(None)
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()