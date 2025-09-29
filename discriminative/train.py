import os
import argparse
import yaml
from easydict import EasyDict as edict

import torch

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from discriminative.utils import get_dataset, get_collate_func, get_compute_loss_func

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def setup_model_and_tokenizer(configs):
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  
    tokenizer.padding_side='right'
    
    model = AutoModelForCausalLM.from_pretrained(
        configs.model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )
    
    if 'lora_config' in configs:
        if configs.training_args.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
            
        lora_config = LoraConfig(**configs.lora_config)
        model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def main(configs):

    os.environ['WANDB_PROJECT'] = args.wandb_project
    configs.training_args["output_dir"] = args.output_dir
    configs.category = args.category

    # model, tokenizer
    model, tokenizer = setup_model_and_tokenizer(configs)
    
    # batch size
    num_gpus = torch.cuda.device_count()
    configs.training_args["per_device_train_batch_size"] = args.per_device_batch_size
    configs.training_args["gradient_accumulation_steps"] = (
        configs.training_args["effective_batch_size"] // \
            (num_gpus*configs.training_args["per_device_train_batch_size"]))
    del configs.training_args["effective_batch_size"]
    
    # trainer
    train_dataset = get_dataset(configs, tokenizer)
    collate_fn = get_collate_func(tokenizer)
    compute_loss_func = get_compute_loss_func(tokenizer)       
    
    training_args = TrainingArguments(**configs.training_args)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
        compute_loss_func=compute_loss_func
    )
    
    # train
    trainer.train()
    
    # save
    if trainer.is_world_process_zero():
        model.save_pretrained(configs.training_args["output_dir"], safe_serialization=True)
        tokenizer.save_pretrained(configs.training_args["output_dir"])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training script for training discriminative reward models')
    
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--wandb_project', type=str, default="Multi-RM")
    parser.add_argument('--per_device_batch_size', type=int, default=4)    
    parser.add_argument("--category", type=str, required=True, choices={
        'law', 'psychology', 'chemistry', 'biology', 'physics', 
        'history', 'economics', 'math', 'business', 'philosophy', 
        'health', 'engineering', 'computer_science', 'other', "all", 'prm800k'
    })
    args = parser.parse_args()

    with open(args.config) as stream:
        try:
            configs = edict(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
                
    main(configs)

