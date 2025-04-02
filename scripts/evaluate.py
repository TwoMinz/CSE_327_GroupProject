"""
Evaluation script for TransCrowd model.
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.config import MODEL_CONFIG, EVAL_CONFIG, DEVICE, CHECKPOINT_DIR, OUTPUT_DIR
from src.data_utils.dataset import get_dataloaders
from src.models.transcrowd import TransCrowd
from src.utils.metrics import evaluate_model
from src.utils.visualization import visualize_model_predictions, visualize_model_performance


def load_model(checkpoint_path, device):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path (str): Path to checkpoint.
        device (torch.device): Device to load model on.

    Returns:
        nn.Module: Loaded model.
    """
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("You need to train the model first or provide a valid checkpoint path.")
        exit(1)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config, with fallback to MODEL_CONFIG
    config = checkpoint.get('config', MODEL_CONFIG)

    # Ensure all required keys are present in config
    for key in ['img_size', 'patch_size', 'in_channels', 'embed_dim', 'depths',
                'num_heads', 'window_size', 'dropout_rate', 'use_checkpoint']:
        if key not in config:
            config[key] = MODEL_CONFIG[key]
            print(f"Warning: Missing '{key}' in checkpoint config, using default value: {MODEL_CONFIG[key]}")

    # Create model
    model = TransCrowd(config)

    # Try loading state dict with error handling
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        print("The checkpoint might be incompatible with the current model architecture.")
        exit(1)

    model = model.to(device)
    model.eval()

    return model


def evaluate(args):
    """
    Evaluate TransCrowd model.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Set up paths
    if args.checkpoint is None:
        # Use latest checkpoint
        experiment_name = args.name
        checkpoint_path = os.path.join(CHECKPOINT_DIR, experiment_name, 'best.pth')
    else:
        checkpoint_path = args.checkpoint

    output_dir = os.path.join(OUTPUT_DIR, 'evaluation', os.path.basename(checkpoint_path).split('.')[0])
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, DEVICE)

    # Get dataloader
    _, test_loader = get_dataloaders(
        part=EVAL_CONFIG['part'],
        batch_size=EVAL_CONFIG['batch_size'],
        num_workers=4,
        pin_memory=True,
        img_size=MODEL_CONFIG['img_size']
    )

    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, DEVICE)

    # Print metrics
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Visualize model performance
    visualize_model_performance(
        metrics,
        save_path=os.path.join(output_dir, 'performance_metrics.png')
    )

    # Visualize model predictions
    print("Generating visualizations...")
    visualize_model_predictions(
        model, test_loader, DEVICE,
        save_dir=os.path.join(output_dir, 'predictions'),
        num_samples=args.num_samples
    )

    print(f"Evaluation completed. Results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate TransCrowd model')
    parser.add_argument('--name', type=str, default=None, help='experiment name')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint')
    parser.add_argument('--num-samples', type=int, default=10, help='number of samples to visualize')
    args = parser.parse_args()

    evaluate(args)