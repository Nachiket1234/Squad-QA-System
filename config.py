"""
Configuration file for BERT-QA training on SQuAD
"""

from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_name: str = 'bert-base-uncased'
    max_length: int = 384
    stride: int = 128
    dropout: float = 0.1
    

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Training
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Optimization
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = True if torch.cuda.is_available() else False
    
    # Logging
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    

@dataclass
class DataConfig:
    """Data paths and preprocessing configuration."""
    train_file: str = 'archive/train-v1.1.json'
    dev_file: str = 'archive/dev-v1.1.json'
    max_length: int = 384
    stride: int = 128
    num_workers: int = 0
    

@dataclass
class PathConfig:
    """Paths for checkpoints, logs, and outputs."""
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    output_dir: str = 'outputs'
    best_model_path: str = 'checkpoints/best_model.pt'
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


class Config:
    """Main configuration class combining all configs."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.paths = PathConfig()
    
    def display(self):
        """Display all configuration parameters."""
        print("\n" + "="*60)
        print("Configuration")
        print("="*60)
        
        print("\n[Model Configuration]")
        for key, value in vars(self.model).items():
            print(f"  {key:20s}: {value}")
        
        print("\n[Training Configuration]")
        for key, value in vars(self.training).items():
            print(f"  {key:20s}: {value}")
        
        print("\n[Data Configuration]")
        for key, value in vars(self.data).items():
            print(f"  {key:20s}: {value}")
        
        print("\n[Path Configuration]")
        for key, value in vars(self.paths).items():
            print(f"  {key:20s}: {value}")
        
        print("="*60 + "\n")
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {
            'model': vars(self.model),
            'training': vars(self.training),
            'data': vars(self.data),
            'paths': vars(self.paths)
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file."""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        
        # Update each config dataclass
        for key, value in config_dict.get('model', {}).items():
            setattr(config.model, key, value)
        
        for key, value in config_dict.get('training', {}).items():
            setattr(config.training, key, value)
        
        for key, value in config_dict.get('data', {}).items():
            setattr(config.data, key, value)
        
        for key, value in config_dict.get('paths', {}).items():
            setattr(config.paths, key, value)
        
        print(f"Configuration loaded from {filepath}")
        return config


# Create default config instance
default_config = Config()


if __name__ == "__main__":
    # Display default configuration
    config = Config()
    config.display()
    
    # Save default configuration
    config.save('config.json')
