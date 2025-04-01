"""
Configuration parameters for the crowd counting project.
"""

import os
import torch
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'ShanghaiTech')
PART_A_DIR = os.path.join(DATA_DIR, 'part_A')
PART_B_DIR = os.path.join(DATA_DIR, 'part_B')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'backbone': 'swin_transformer_v2',  # Options: 'vit', 'swin_transformer', 'swin_transformer_v2'
    'img_size': 384,                    # Input image size
    'patch_size': 16,                   # Patch size for the transformer
    'in_channels': 3,                   # Number of input channels
    'embed_dim': 128,                   # Embedding dimension
    'depths': [2, 2, 18, 2],            # Depths of each Swin Transformer stage
    'num_heads': [4, 8, 16, 32],        # Number of attention heads in different layers
    'window_size': 12,                  # Window size for Swin Transformer
    'dropout_rate': 0.2,                # Dropout rate
    'use_checkpoint': True,            # Whether to use checkpointing to save memory
}

# Training configuration
TRAIN_CONFIG = {
    'part': 'A',                        # Dataset part to use: 'A' or 'B'
    'batch_size': 32,                   # Batch size for training
    'num_epochs': 150,                  # Maximum number of epochs
    'learning_rate': 1e-4,              # Initial learning rate
    'weight_decay': 1e-4,               # Weight decay
    'lr_scheduler': 'cosine',           # Learning rate scheduler: 'step', 'cosine'
    'lr_decay_epochs': [40, 80],        # Epochs at which to decay learning rate (if using 'step')
    'lr_decay_rate': 0.1,               # Learning rate decay factor (if using 'step')
    'warmup_epochs': 10,                 # Number of warmup epochs
    'early_stopping_patience': 10,      # Patience for early stopping
    'grad_clip_norm': 1.0,              # Gradient clipping norm
    'seed': 42,                         # Random seed for reproducibility
    'num_workers': 4,                   # Number of data loading workers
    'pin_memory': True,                 # Pin memory for faster data transfer to GPU
    'save_freq': 5,                     # Checkpoint saving frequency (epochs)
    'log_freq': 10,                     # Logging frequency (iterations)
    'val_freq': 1,                      # Validation frequency (epochs)
}

# Evaluation configuration
EVAL_CONFIG = {
    'part': 'A',                        # Dataset part to use for evaluation
    'batch_size': 16,                   # Batch size for evaluation
    'metrics': ['mae', 'mse'],          # Metrics to use for evaluation
    'checkpoint_path': None,            # Path to the checkpoint to load (None for latest)
}

# Experiment name (used for logging and checkpointing)
def get_experiment_name():
    """Generate a unique experiment name based on configuration and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backbone = MODEL_CONFIG['backbone']
    part = TRAIN_CONFIG['part']
    return f"{backbone}_part{part}_{timestamp}"

EXPERIMENT_NAME = get_experiment_name()

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')