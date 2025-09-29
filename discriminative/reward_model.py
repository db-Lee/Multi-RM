import torch
import math
import statistics
from typing import Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM

Device = Union[str, torch.device]

def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  
    tokenizer.padding_side = 'left' 
    tokenizer.truncation_side = 'left'
    return tokenizer

class RewardModel:    
    def __init__(
        self,
        aggregation: str = 'full',
        device: Optional[Device] = None,
        model_id: str = 'dongboklee/dORM-14B'
    ) -> None:
        self.device = (
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        self.tokenizer = get_tokenizer(model_id)
            
        self.candidate_tokens = [
            self.tokenizer.encode("-", add_special_tokens=False)[-1], 
            self.tokenizer.encode("+", add_special_tokens=False)[-1]
        ]
        self.tag_id = self.tokenizer.encode(" \n\n\n\n", add_special_tokens=False)[-1]
        
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()
        self.model.to(self.device)
        
        self.aggregation = aggregation

    def __call_single(self, single_beam: str) -> list[float]:
        '''
        Computes scores for each reasoning step in the single_beam.

        Args:
            single_beam (str): A single reasoning beam, consisting of Question + Solution.

        Returns:
            list[float]: The scores for each step in the Solution.
        '''
        input_id = torch.tensor([self.tokenizer.encode(single_beam)]).to(self.device)

        with torch.no_grad():
            logits = self.model(input_id).logits[:,:,self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,1] 
            step_scores = scores[input_id == self.tag_id]
            step_probs = step_scores.tolist()

        return self._aggregate_scores(step_probs)

    def __call_batch(self, steps: list[str]) -> list[list]:
        '''
        Computes scores for a batch of reasoning beams efficiently.

        Args:
            steps (list[str]): A list of reasoning beams.

        Returns:
            list[list]: A list of step scores for each beam.
        '''
        # Tokenize all inputs
        encoded_inputs = self.tokenizer(
            steps, 
            return_tensors='pt', 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        input_ids = encoded_inputs['input_ids']  # Shape: [batch_size, seq_len]
        attention_mask = encoded_inputs['attention_mask']

        with torch.no_grad():
            # Get logits for candidate tokens only
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :, self.candidate_tokens]  # [batch_size, seq_len, 2]
            scores = logits.softmax(dim=-1)[:, :, 1]  # [batch_size, seq_len] - probability of "+"
            
        results = []
        for i in range(input_ids.size(0)):
            # Find positions where tag_id appears for this sequence
            tag_positions = (input_ids[i] == self.tag_id).nonzero(as_tuple=True)[0]
            
            if len(tag_positions) > 0:
                step_scores = scores[i, tag_positions].tolist()
                results.append(self._aggregate_scores(step_scores))
            else:
                # No tags found, return empty or default
                results.append([] if self.aggregation == 'full' else 0.0)
        
        return results

    def _aggregate_scores(self, step_probs: list[float]) -> Union[float, list[float]]:
        '''Helper method to aggregate step probabilities based on aggregation method'''
        if not step_probs:
            return [] if self.aggregation == 'full' else 0.0
            
        if self.aggregation == 'min':
            return min(step_probs)
        elif self.aggregation == 'max':
            return max(step_probs)
        elif self.aggregation == 'mean':
            return statistics.mean(step_probs)
        elif self.aggregation == 'prod':
            return math.prod(step_probs)
        elif self.aggregation == 'last':
            return step_probs[-1]
        elif self.aggregation == 'full':
            return step_probs
        else:
            raise NotImplementedError(f"Aggregation method '{self.aggregation}' not implemented")

    def __call__(self, steps: list[str]) -> list:
        '''
        Computes scores for a list of reasoning beams using batch processing.

        Args:
            steps (list[str]): A list of reasoning beams.

        Returns:
            list: A list of step scores for each beam.
        '''
        if len(steps) == 1:
            # Use single processing for single item to maintain exact compatibility
            return [self.__call_single(steps[0])]
        else:
            # Use batch processing for multiple items
            return self.__call_batch(steps)