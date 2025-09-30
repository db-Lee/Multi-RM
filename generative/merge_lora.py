import os
import json
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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