import os
import re
import json
import math
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from generative.prompt_formats import CHAT_TEMPLATE, ORM_PROMPT_FORMAT, PRM_PROMPT_FORMAT
from datasets import Dataset

class RewardModel:
    def __init__(
            self, 
            process_id, 
            gpu_ids, 
            model_id="dongboklee/gORM-14B",
            task_type="gORM",
            tensor_parallel_size=1,
            n_generation=10,
            temperature=0.6,
            max_tokens=8192,
            top_p=1,
            top_k=-1,
            min_p=0,
            logprobs=20,
            batch_size=1
        ):
        
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))        
        self.process_id = process_id
        self.batch_size = batch_size
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.chat_template = CHAT_TEMPLATE
        
        self.yes_id = self.tokenizer.encode(" Yes", add_special_tokens=False)[-1]
        self.no_id = self.tokenizer.encode(" No", add_special_tokens=False)[-1]
        
        print(f"Process {process_id} (GPUs {gpu_ids}): Initializing vLLM...")
        
        if task_type == "gORM":
            self.stop_sequences = [
                "Verification: Is the answer correct (Yes/No)? Yes",
                "Verification: Is the answer correct (Yes/No)? No"
            ]
        else:
            self.stop_sequences = [
                "Is the solution correct? Yes",
                "Is the solution correct? No"
            ]
        
        self.llm = LLM(
            model=model_id,
            tokenizer=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.95,
            trust_remote_code=True
        )
        print(f"Process {process_id}: Model ready")
        
        # Set up generation parameters with stop sequences
        self.sampling_params = SamplingParams(
            n=n_generation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            logprobs=logprobs,
            stop=self.stop_sequences,
            include_stop_str_in_output=True
        )
        
        self.prompt_format = ORM_PROMPT_FORMAT if task_type == "gORM" else PRM_PROMPT_FORMAT
    
    def process_batch(self, category: str, dataset: Dataset):
        """Process a dataset batch for a specific category."""
        processed_results = []
        
        for batch_start in tqdm(range(0, len(dataset), self.batch_size), 
                               desc=f"P{self.process_id}-{category}"):
            batch_end = min(batch_start + self.batch_size, len(dataset))
            batch_data = dataset.select(range(batch_start, batch_end))
            
            # Prepare prompts
            batch_prompts = []
            for data in batch_data:
                for cot in data["cots"]:
                    prompt = self.prompt_format(category, data["question"], cot)
                    prompt = self.tokenizer.apply_chat_template(
                        [{'role': "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True, add_special_tokens=False
                    ) + "Let's verify step by step:"
                    
                    # Remove BOS token if present (vLLM adds it)
                    if self.tokenizer.bos_token and prompt.startswith(self.tokenizer.bos_token):
                        prompt = prompt[len(self.tokenizer.bos_token):]
                    
                    batch_prompts.append(prompt)
            
            # Generate
            outputs = self.llm.generate(
                batch_prompts, 
                self.sampling_params,
                use_tqdm=False
            )
            
            # Process outputs
            output_idx = 0
            for data in batch_data:
                result = {
                    "q_id": data["q_id"],
                    "cot_ids": data["cot_ids"],
                    "rewards": []
                }
                
                for _ in data["cots"]:
                    rewards = []
                    
                    for completion in outputs[output_idx].outputs:
                        # if stop sequence does not exist
                        critique = completion.text
                        if not any(seq in critique for seq in self.stop_sequences):
                            rewards.append(np.nan)
                            continue
                                              
                        # Get logprobs for the Yes/No token
                        # With stop sequences and include_stop_str_in_output=True, 
                        # the Yes/No token should be at [-1] position
                        if completion.logprobs and len(completion.logprobs) > 0:
                            # Use [-1] since stop sequence tokens are included
                            lp_dict = {k: float(v.logprob) for k, v in completion.logprobs[-1].items()}
                            
                            # Calculate reward
                            yes_lp = lp_dict.get(self.yes_id, None)
                            no_lp = lp_dict.get(self.no_id, None)
                            
                            if yes_lp is not None and no_lp is not None:
                                exp_yes, exp_no = math.exp(yes_lp), math.exp(no_lp)
                                rewards.append(exp_yes / (exp_yes + exp_no))
                            elif yes_lp is not None and no_lp is None:
                                rewards.append(1.0)
                            elif yes_lp is None and no_lp is not None:
                                rewards.append(0.0)
                            else:
                                rewards.append(np.nan)
                        else:
                            rewards.append(np.nan)
                    
                    result["rewards"].append(rewards)
                    output_idx += 1
                
                processed_results.append(result)
        
        return processed_results