"""
Training script for BERT Question Answering on SQuAD
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.dataset import SQuADDataset
from data.dataloader import create_squad_dataloaders, squad_collate_fn


class BertQATrainer:
    """Trainer class for BERT Question Answering."""
    
    def __init__(self, config: Config):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.training.device)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model, tokenizer, datasets
        self.tokenizer = None
        self.model = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        
        # Mixed precision training
        self.scaler = None
        if config.training.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.logger.info(f"Trainer initialized on device: {self.device}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = Path(self.config.paths.log_dir) / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging to {log_file}")
    
    def setup(self):
        """Setup model, datasets, and optimizer."""
        self.logger.info("Setting up training components...")
        
        # Load tokenizer
        self.logger.info(f"Loading tokenizer: {self.config.model.model_name}")
        self.tokenizer = BertTokenizerFast.from_pretrained(self.config.model.model_name)
        
        # Load datasets
        self.logger.info("Loading datasets...")
        train_dataset = SQuADDataset(
            self.config.data.train_file,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_length,
            stride=self.config.data.stride
        )
        
        eval_dataset = SQuADDataset(
            self.config.data.dev_file,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_length,
            stride=self.config.data.stride
        )
        
        # Create dataloaders
        self.train_dataloader, self.eval_dataloader = create_squad_dataloaders(
            train_dataset,
            eval_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.data.num_workers,
            collate_fn=squad_collate_fn
        )
        
        self.logger.info(f"Training batches: {len(self.train_dataloader)}")
        self.logger.info(f"Evaluation batches: {len(self.eval_dataloader)}")
        
        # Initialize model
        self.logger.info(f"Loading model: {self.config.model.model_name}")
        self.model = BertForQuestionAnswering.from_pretrained(self.config.model.model_name)
        self.model.to(self.device)
        
        # Setup optimizer
        self.setup_optimizer()
        
        self.logger.info("Setup complete!")
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Prepare optimizer parameters
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.training.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        # Initialize optimizer
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.learning_rate,
            eps=self.config.training.adam_epsilon,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2)
        )
        
        # Calculate total training steps
        num_training_steps = len(self.train_dataloader) * self.config.training.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.training.warmup_ratio)
        
        # Initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.logger.info(f"Total training steps: {num_training_steps}")
        self.logger.info(f"Warmup steps: {num_warmup_steps}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.config.training.mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch['token_type_ids'],
                        start_positions=batch['start_positions'],
                        end_positions=batch['end_positions']
                    )
                    loss = outputs.loss
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids'],
                    start_positions=batch['start_positions'],
                    end_positions=batch['end_positions']
                )
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Logging
            if self.global_step % self.config.training.logging_steps == 0:
                avg_loss = total_loss / (batch_idx + 1)
                self.logger.info(
                    f"Epoch: {self.current_epoch + 1}, "
                    f"Step: {self.global_step}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Avg Loss: {avg_loss:.4f}, "
                    f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                )
            
            # Save checkpoint
            if self.global_step % self.config.training.save_steps == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
        
        avg_epoch_loss = total_loss / len(self.train_dataloader)
        return avg_epoch_loss
    
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.eval_dataloader, desc="Evaluating")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids'],
                    start_positions=batch['start_positions'],
                    end_positions=batch['end_positions']
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.eval_dataloader)
        return avg_loss
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.config.display()
        
        # Setup if not already done
        if self.model is None:
            self.setup()
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self.train_epoch()
            self.logger.info(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}")
            
            # Evaluate
            eval_loss = self.evaluate()
            self.logger.info(f"Epoch {epoch + 1} - Evaluation Loss: {eval_loss:.4f}")
            
            # Check for improvement
            if eval_loss < self.best_eval_loss - self.config.training.early_stopping_threshold:
                self.best_eval_loss = eval_loss
                self.patience_counter = 0
                self.save_checkpoint(self.config.paths.best_model_path, is_best=True)
                self.logger.info(f"New best model saved with eval loss: {eval_loss:.4f}")
            else:
                self.patience_counter += 1
                self.logger.info(f"No improvement. Patience: {self.patience_counter}/{self.config.training.early_stopping_patience}")
            
            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save epoch checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        self.logger.info("Training complete!")
        self.logger.info(f"Best evaluation loss: {self.best_eval_loss:.4f}")
    
    def save_checkpoint(self, filename, is_best=False):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.paths.checkpoint_dir) / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_eval_loss': self.best_eval_loss,
            'config': self.config.to_dict()
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            self.logger.info(f"Best model saved to {checkpoint_path}")
        else:
            self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_eval_loss = checkpoint['best_eval_loss']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")


def main():
    """Main training function."""
    # Load configuration
    config = Config()
    
    # Create trainer
    trainer = BertQATrainer(config)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
