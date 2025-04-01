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
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config
    config = checkpoint.get('config', MODEL_CONFIG)

    # Create model
    model = TransCrowd(config)
    model.load_state_dict(checkpoint['model_state_dict'])
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