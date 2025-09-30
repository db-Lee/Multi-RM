import re
import numpy as np
from datasets import Dataset

def parse_orm_label(text):
    pattern = r'Verification: Is the answer correct \(Yes/No\)\?\s*(Yes|No)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        verdict = match.group(1).lower()
        if verdict == 'yes':
            return 1
        elif verdict == 'no':
            return -1
        else:
            return np.nan
    return np.nan

def parse_prm_label(text):
    # Match literal: The step is \\boxed{correct}
    pattern = r'The step is \\boxed{(correct|incorrect)}'
    verdicts = re.findall(pattern, text, re.IGNORECASE)
    array = []
    for v in verdicts:
        if v.lower() == 'correct':
            array.append(1)
        elif v.lower() == 'incorrect':
            array.append(-1)
        else:
            array.append(np.nan)
    return array

def trim_after_first_verdict(text):
    pattern = r'Verification: Is the answer correct \(Yes/No\)\?\s*(Yes|No)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return text[:match.end()]
    else:
        return text  # or "" or None if you prefer to indicate no match

def truncate_after_last_boxed_step(text):
    pattern = r'The step is \\boxed{(correct|incorrect)}'
    matches = list(re.finditer(pattern, text))

    if not matches:
        return text  # nothing to truncate

    last_match = matches[-1]
    end_index = last_match.end()  # keep the full match
    return text[:end_index]

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