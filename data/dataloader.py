"""
DataLoader utilities for SQuAD Question Answering
"""

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, default_data_collator
from typing import Optional, Callable
from .dataset import SQuADDataset


def create_squad_dataloaders(
    train_dataset: SQuADDataset,
    eval_dataset: SQuADDataset,
    batch_size: int = 16,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None
):
    """
    Create DataLoaders for training and evaluation.
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        batch_size: Batch size for training and evaluation
        num_workers: Number of workers for data loading
        collate_fn: Custom collate function (default uses transformers' collator)
        
    Returns:
        Tuple of (train_dataloader, eval_dataloader)
    """
    if collate_fn is None:
        collate_fn = default_data_collator
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_dataloader, eval_dataloader


def squad_collate_fn(batch):
    """
    Custom collate function for SQuAD data.
    
    Handles batching of tokenized examples with proper padding.
    
    Args:
        batch: List of examples from the dataset
        
    Returns:
        Batched dictionary with tensors
    """
    # Stack all tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    start_positions = torch.stack([item['start_positions'] for item in batch])
    end_positions = torch.stack([item['end_positions'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'start_positions': start_positions,
        'end_positions': end_positions
    }


class SquadDataModule:
    """
    Data module for SQuAD that encapsulates all data loading logic.
    
    This provides a clean interface for managing datasets and dataloaders.
    """
    
    def __init__(
        self,
        train_path: str,
        dev_path: str,
        model_name: str = 'bert-base-uncased',
        max_length: int = 384,
        stride: int = 128,
        batch_size: int = 16,
        num_workers: int = 0
    ):
        """
        Initialize SQuAD data module.
        
        Args:
            train_path: Path to training JSON file
            dev_path: Path to dev JSON file
            model_name: Pretrained model name for tokenizer
            max_length: Maximum sequence length
            stride: Stride for long contexts
            batch_size: Batch size
            num_workers: Number of data loading workers
        """
        self.train_path = train_path
        self.dev_path = dev_path
        self.model_name = model_name
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        
        # Will be initialized in setup()
        self.train_dataset = None
        self.eval_dataset = None
        self.train_dataloader = None
        self.eval_dataloader = None
    
    def setup(self):
        """Load datasets and create dataloaders."""
        print("Setting up datasets...")
        
        # Create datasets
        self.train_dataset = SQuADDataset(
            self.train_path,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            stride=self.stride
        )
        
        self.eval_dataset = SQuADDataset(
            self.dev_path,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            stride=self.stride
        )
        
        # Create dataloaders
        self.train_dataloader, self.eval_dataloader = create_squad_dataloaders(
            self.train_dataset,
            self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=squad_collate_fn
        )
        
        print(f"✓ Setup complete")
        print(f"  Training batches: {len(self.train_dataloader)}")
        print(f"  Evaluation batches: {len(self.eval_dataloader)}")
    
    def get_train_dataloader(self):
        """Get training dataloader."""
        if self.train_dataloader is None:
            self.setup()
        return self.train_dataloader
    
    def get_eval_dataloader(self):
        """Get evaluation dataloader."""
        if self.eval_dataloader is None:
            self.setup()
        return self.eval_dataloader
    
    def get_datasets(self):
        """Get both datasets."""
        if self.train_dataset is None or self.eval_dataset is None:
            self.setup()
        return self.train_dataset, self.eval_dataset
    
    def print_statistics(self):
        """Print dataset statistics."""
        if self.train_dataset is None:
            self.setup()
        
        train_stats = self.train_dataset.get_statistics()
        eval_stats = self.eval_dataset.get_statistics()
        
        print("\n" + "="*60)
        print("Dataset Statistics")
        print("="*60)
        print("\nTraining Set:")
        for key, value in train_stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        print("\nEvaluation Set:")
        for key, value in eval_stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        print("="*60)


def validate_batch(batch, tokenizer):
    """
    Validate a batch of data by decoding and checking answer spans.
    
    Args:
        batch: Batch from dataloader
        tokenizer: Tokenizer used for encoding
        
    Returns:
        List of validation results
    """
    results = []
    
    batch_size = batch['input_ids'].shape[0]
    
    for i in range(batch_size):
        input_ids = batch['input_ids'][i]
        start_pos = batch['start_positions'][i].item()
        end_pos = batch['end_positions'][i].item()
        
        # Decode full text
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        
        # Decode answer span
        if start_pos > 0 and end_pos > 0:
            answer_ids = input_ids[start_pos:end_pos+1]
            answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
        else:
            answer_text = "[NO ANSWER]"
        
        results.append({
            'full_text': full_text[:200] + "...",
            'answer': answer_text,
            'start_pos': start_pos,
            'end_pos': end_pos
        })
    
    return results
