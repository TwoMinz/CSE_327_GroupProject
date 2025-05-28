"""
Script to compare different backbone architectures for TransCrowd model.
"""

import os
import argparse
import time
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from src.config import MODEL_CONFIG, TRAIN_CONFIG, EVAL_CONFIG, DEVICE, OUTPUT_DIR
from src.data_utils.dataset import get_dataloaders
from src.models.transcrowd import TransCrowd
from src.utils.metrics import evaluate_model


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, dataloader, device, num_batches=10):
    """Measure average inference time per batch."""
    model.eval()
    times = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            images = batch['image'].to(device)

            # Warm up GPU
            if i == 0:
                _ = model(images)
                torch.cuda.synchronize() if torch.cuda.is_available() else None

            start_time = time.time()
            _ = model(images)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            times.append(end_time - start_time)

    return np.mean(times), np.std(times)


def compare_backbones(backbones, save_dir=None):
    """
    Compare different backbone architectures.

    Args:
        backbones (list): List of backbone configurations to compare
        save_dir (str): Directory to save results
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Get dataloader
    _, test_loader = get_dataloaders(
        part=EVAL_CONFIG['part'],
        batch_size=EVAL_CONFIG['batch_size'],
        num_workers=4,
        pin_memory=True,
        img_size=MODEL_CONFIG['img_size']
    )

    results = []

    for backbone_config in backbones:
        print(f"\n{'=' * 50}")
        print(f"Testing backbone: {backbone_config['name']}")
        print(f"{'=' * 50}")

        # Update model config
        config = MODEL_CONFIG.copy()
        config.update(backbone_config['config'])

        try:
            # Create model
            model = TransCrowd(config)
            model = model.to(DEVICE)

            # Count parameters
            num_params = count_parameters(model)
            print(f"Number of parameters: {num_params:,}")

            # Measure inference time
            print("Measuring inference time...")
            avg_time, std_time = measure_inference_time(model, test_loader, DEVICE)
            print(f"Average inference time: {avg_time:.4f} ± {std_time:.4f} seconds per batch")

            # Evaluate model (if checkpoint exists)
            checkpoint_path = backbone_config.get('checkpoint_path')
            if checkpoint_path and os.path.exists(checkpoint_path):
                print("Loading checkpoint and evaluating...")
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])

                metrics = evaluate_model(model, test_loader, DEVICE)
                print("Evaluation metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")
            else:
                print("No checkpoint found, skipping evaluation...")
                metrics = {'mae': None, 'mse': None, 'rmse': None, 'mape': None}

            # Store results
            result = {
                'backbone': backbone_config['name'],
                'num_parameters': num_params,
                'avg_inference_time': avg_time,
                'std_inference_time': std_time,
                **metrics
            }
            results.append(result)

        except Exception as e:
            print(f"Error testing {backbone_config['name']}: {str(e)}")
            result = {
                'backbone': backbone_config['name'],
                'num_parameters': None,
                'avg_inference_time': None,
                'std_inference_time': None,
                'mae': None,
                'mse': None,
                'rmse': None,
                'mape': None,
                'error': str(e)
            }
            results.append(result)

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Save results
    if save_dir:
        df_results.to_csv(os.path.join(save_dir, 'backbone_comparison.csv'), index=False)
        print(f"\nResults saved to {os.path.join(save_dir, 'backbone_comparison.csv')}")

    # Print comparison table
    print(f"\n{'=' * 80}")
    print("BACKBONE COMPARISON RESULTS")
    print(f"{'=' * 80}")
    print(df_results.to_string(index=False))

    # Create visualizations
    if save_dir:
        create_comparison_plots(df_results, save_dir)

    return df_results


def create_comparison_plots(df_results, save_dir):
    """Create comparison plots for different metrics."""

    # Filter out failed experiments
    df_valid = df_results.dropna(subset=['num_parameters', 'avg_inference_time'])

    if len(df_valid) == 0:
        print("No valid results to plot")
        return

    # 1. Parameters vs Inference Time
    plt.figure(figsize=(10, 6))
    plt.scatter(df_valid['num_parameters'] / 1e6, df_valid['avg_inference_time'],
                s=100, alpha=0.7)

    for i, row in df_valid.iterrows():
        plt.annotate(row['backbone'],
                     (row['num_parameters'] / 1e6, row['avg_inference_time']),
                     xytext=(5, 5), textcoords='offset points')

    plt.xlabel('Number of Parameters (Millions)')
    plt.ylabel('Average Inference Time (seconds)')
    plt.title('Model Efficiency: Parameters vs Inference Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'parameters_vs_time.png'))
    plt.close()

    # 2. MAE comparison (if available)
    df_mae = df_valid.dropna(subset=['mae'])
    if len(df_mae) > 0:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_mae['backbone'], df_mae['mae'])
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title('Model Accuracy Comparison')
        plt.xticks(rotation=45, ha='right')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'mae_comparison.png'))
        plt.close()

    # 3. Efficiency Score (lower MAE, fewer parameters, faster inference is better)
    if len(df_mae) > 0:
        # Normalize metrics (0-1 scale)
        df_score = df_mae.copy()

        # For MAE: lower is better, so we use 1 - normalized_mae
        mae_norm = (df_score['mae'] - df_score['mae'].min()) / (df_score['mae'].max() - df_score['mae'].min())
        mae_score = 1 - mae_norm

        # For parameters: fewer is better
        param_norm = (df_score['num_parameters'] - df_score['num_parameters'].min()) / (
                    df_score['num_parameters'].max() - df_score['num_parameters'].min())
        param_score = 1 - param_norm

        # For inference time: faster is better
        time_norm = (df_score['avg_inference_time'] - df_score['avg_inference_time'].min()) / (
                    df_score['avg_inference_time'].max() - df_score['avg_inference_time'].min())
        time_score = 1 - time_norm

        # Combined efficiency score (equal weights)
        efficiency_score = (mae_score + param_score + time_score) / 3

        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_score['backbone'], efficiency_score)
        plt.ylabel('Efficiency Score (0-1, higher is better)')
        plt.title('Overall Efficiency Score\n(Combines Accuracy, Parameter Count, and Speed)')
        plt.xticks(rotation=45, ha='right')

        # Add value labels on bars
        for bar, score in zip(bars, efficiency_score):
            height = bar.get_height()
            plt.annotate(f'{score:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'efficiency_score.png'))
        plt.close()

    print(f"Comparison plots saved to {save_dir}")


def main():
    """Main function to run backbone comparison."""
    parser = argparse.ArgumentParser(description='Compare different backbone architectures')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results')
    args = parser.parse_args()

    # Define backbone configurations to compare
    backbones = [
        {
            'name': 'DINOv2-Small',
            'config': {
                'backbone': 'dinov2',
                'dinov2_size': 'small',
                'pretrained': True,
                'freeze_backbone': False
            },
            'checkpoint_path': os.path.join(OUTPUT_DIR, 'checkpoints', 'dinov2_small', 'best.pth')
        },
        {
            'name': 'DINOv2-Base',
            'config': {
                'backbone': 'dinov2',
                'dinov2_size': 'base',
                'pretrained': True,
                'freeze_backbone': False
            },
            'checkpoint_path': os.path.join(OUTPUT_DIR, 'checkpoints', 'dinov2_base', 'best.pth')
        },
        {
            'name': 'DINOv2-Large',
            'config': {
                'backbone': 'dinov2',
                'dinov2_size': 'large',
                'pretrained': True,
                'freeze_backbone': False
            },
            'checkpoint_path': os.path.join(OUTPUT_DIR, 'checkpoints', 'dinov2_large', 'best.pth')
        },
        {
            'name': 'Swin-Transformer-V2',
            'config': {
                'backbone': 'swin_transformer_v2',
                'embed_dim': 128,
                'depths': [2, 6, 6, 2],
                'num_heads': [4, 8, 16, 32],
                'window_size': 12
            },
            'checkpoint_path': os.path.join(OUTPUT_DIR, 'checkpoints', 'swin_v2', 'best.pth')
        }
    ]

    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, 'backbone_comparison')
    else:
        output_dir = args.output_dir

    # Run comparison
    results = compare_backbones(backbones, output_dir)

    print(f"\nBackbone comparison completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()