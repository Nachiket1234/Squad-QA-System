"""
SQuAD Dataset Parser and PyTorch Dataset Class
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple


class SQuADDataset(Dataset):
    """
    PyTorch Dataset class for SQuAD v1.1 format.
    
    Parses JSON files and returns question-context-answer tuples.
    """
    
    def __init__(self, json_path: str, tokenizer=None, max_length: int = 384, 
                 stride: int = 128, include_impossible: bool = False):
        """
        Initialize SQuAD dataset.
        
        Args:
            json_path: Path to SQuAD JSON file (train-v1.1.json or dev-v1.1.json)
            tokenizer: Hugging Face tokenizer (optional, for preprocessing)
            max_length: Maximum sequence length for tokenization
            stride: Stride for handling long contexts
            include_impossible: Whether to include impossible questions (SQuAD 2.0 feature)
        """
        self.json_path = Path(json_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.include_impossible = include_impossible
        
        # Parse the JSON file
        self.examples = self._load_and_parse()
        
        print(f"Loaded {len(self.examples)} examples from {self.json_path.name}")
    
    def _load_and_parse(self) -> List[Dict]:
        """
        Load and parse SQuAD JSON file.
        
        Returns:
            List of dictionaries containing question-context-answer information
        """
        with open(self.json_path, 'r', encoding='utf-8') as f:
            squad_data = json.load(f)
        
        examples = []
        
        for article in squad_data['data']:
            title = article['title']
            
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                
                for qa in paragraph['qas']:
                    question_id = qa['id']
                    question = qa['question']
                    
                    # Handle SQuAD 2.0 impossible questions
                    is_impossible = qa.get('is_impossible', False)
                    
                    if is_impossible and not self.include_impossible:
                        continue
                    
                    # Extract answers
                    if is_impossible:
                        # No answer for impossible questions
                        examples.append({
                            'id': question_id,
                            'title': title,
                            'question': question,
                            'context': context,
                            'answer_text': '',
                            'answer_start': -1,
                            'is_impossible': True
                        })
                    else:
                        # Use the first answer (SQuAD v1.1 has multiple answers per question)
                        answer = qa['answers'][0] if qa['answers'] else None
                        
                        if answer:
                            examples.append({
                                'id': question_id,
                                'title': title,
                                'question': question,
                                'context': context,
                                'answer_text': answer['text'],
                                'answer_start': answer['answer_start'],
                                'is_impossible': False
                            })
        
        return examples
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single example.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary containing the example data (optionally tokenized)
        """
        example = self.examples[idx]
        
        # If tokenizer is provided, return tokenized version
        if self.tokenizer:
            return self._tokenize_example(example)
        
        # Otherwise return raw example
        return example
    
    def _tokenize_example(self, example: Dict) -> Dict:
        """
        Tokenize a single example and convert answer position to token indices.
        
        Args:
            example: Dictionary containing question, context, and answer
            
        Returns:
            Dictionary with tokenized inputs and answer token positions
        """
        question = example['question']
        context = example['context']
        answer_start_char = example['answer_start']
        answer_text = example['answer_text']
        
        # Tokenize with offset mappings
        encoding = self.tokenizer(
            question,
            context,
            truncation='only_second',  # Only truncate context
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Find answer span in tokens
        start_token_idx, end_token_idx = self._find_answer_tokens(
            answer_start_char,
            answer_text,
            encoding['offset_mapping'][0],
            encoding.sequence_ids(0)
        )
        
        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'token_type_ids': encoding['token_type_ids'][0],
            'start_positions': torch.tensor(start_token_idx if start_token_idx is not None else 0),
            'end_positions': torch.tensor(end_token_idx if end_token_idx is not None else 0),
            'id': example['id'],
            'question': question,
            'context': context,
            'answer_text': answer_text
        }
    
    def _find_answer_tokens(self, answer_start_char: int, answer_text: str, 
                           offset_mapping, sequence_ids) -> Tuple[int, int]:
        """
        Convert character-level answer position to token-level positions.
        
        Args:
            answer_start_char: Character position where answer starts
            answer_text: The answer text
            offset_mapping: Tensor of (start, end) character offsets for each token
            sequence_ids: List indicating which tokens belong to context
            
        Returns:
            Tuple of (start_token_idx, end_token_idx)
        """
        # Handle impossible questions
        if answer_start_char == -1:
            return None, None
        
        answer_end_char = answer_start_char + len(answer_text)
        
        # Find start token
        start_token_idx = None
        for idx, (start, end) in enumerate(offset_mapping):
            # Only consider context tokens (sequence_id == 1)
            if sequence_ids[idx] == 1:
                if start <= answer_start_char < end:
                    start_token_idx = idx
                    break
        
        # Find end token
        end_token_idx = None
        for idx, (start, end) in enumerate(offset_mapping):
            if sequence_ids[idx] == 1:
                if start < answer_end_char <= end:
                    end_token_idx = idx
                    break
        
        # If answer not found in this chunk (due to truncation), return None
        if start_token_idx is None or end_token_idx is None:
            return None, None
        
        return start_token_idx, end_token_idx
    
    def get_raw_examples(self) -> List[Dict]:
        """Return all raw examples (without tokenization)."""
        return self.examples
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'num_examples': len(self.examples),
            'avg_context_length': sum(len(ex['context']) for ex in self.examples) / len(self.examples),
            'avg_question_length': sum(len(ex['question']) for ex in self.examples) / len(self.examples),
            'avg_answer_length': sum(len(ex['answer_text']) for ex in self.examples if ex['answer_text']) / 
                                 len([ex for ex in self.examples if ex['answer_text']]),
            'num_impossible': sum(1 for ex in self.examples if ex.get('is_impossible', False))
        }
        return stats


def load_squad_data(train_path: str, dev_path: str, tokenizer=None, 
                    max_length: int = 384, stride: int = 128) -> Tuple[SQuADDataset, SQuADDataset]:
    """
    Convenience function to load both training and dev datasets.
    
    Args:
        train_path: Path to training JSON file
        dev_path: Path to dev JSON file
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        stride: Stride for long contexts
        
    Returns:
        Tuple of (train_dataset, dev_dataset)
    """
    train_dataset = SQuADDataset(train_path, tokenizer, max_length, stride)
    dev_dataset = SQuADDataset(dev_path, tokenizer, max_length, stride)
    
    return train_dataset, dev_dataset
