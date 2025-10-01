import torch.nn.functional as F
from transformers import DataCollatorForTokenClassification
from datasets import Dataset

from discriminative.data import TokenizedDataset

def get_dataset(configs, tokenizer):
    
    train_dataset = TokenizedDataset(
        data_path=configs.train_data_path, 
        category=configs.category,
        tokenizer=tokenizer,
        label_mask_token_id=-100,
        label_last_n=1,
        max_length=configs.max_length if 'max_length' in configs else None,
        task_type=configs.task_type
    )
    return train_dataset

def get_collate_func(tokenizer):      
    return DataCollatorForTokenClassification(
        tokenizer=tokenizer, 
        padding='longest', 
        label_pad_token_id=-100,
        return_tensors='pt'
    )

def get_compute_loss_func(tokenizer):
    candidate_tokens = [
        tokenizer.encode("-", add_special_tokens=False)[-1], 
        tokenizer.encode("+", add_special_tokens=False)[-1]
    ]
      
    def compute_loss_func(outputs, labels, num_items_in_batch):
        logits = outputs.logits[:,:,candidate_tokens].reshape(-1,2)

        if num_items_in_batch is None:
            loss = F.cross_entropy(input=logits,
                            target=labels.flatten(),
                            ignore_index=-100)
            return loss
        
        loss = F.cross_entropy(input=logits,
                            target=labels.flatten(),
                            ignore_index=-100,
                            reduction='sum')

        return loss / num_items_in_batch    
    return compute_loss_func

def split_dataset_for_gpus(dataset: Dataset, num_gpus: int):
    batch_size = len(dataset) // num_gpus
    
    batches = []
    for i in range(num_gpus):
        start_idx = i * batch_size
        if i == num_gpus - 1:  # Last GPU gets remaining items
            end_idx = len(dataset)
        else:
            end_idx = (i + 1) * batch_size
        
        batch_data = dataset.select(range(start_idx, end_idx))        
        batches.append(batch_data)
    
    return batches