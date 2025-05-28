"""
Visualization utilities for crowd counting models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import torch
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter

from src.config import OUTPUT_DIR


def denormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Denormalize image tensor to RGB image.

    Args:
        tensor (torch.Tensor): Image tensor with shape (C, H, W).
        mean (tuple): Mean used for normalization.
        std (tuple): Standard deviation used for normalization.

    Returns:
        numpy.ndarray: Denormalized image with shape (H, W, C) in range [0, 1].
    """
    # Convert to numpy
    tensor = tensor.cpu().numpy()

    # Denormalize
    for i in range(3):
        tensor[i] = tensor[i] * std[i] + mean[i]

    # Clip to [0, 1]
    tensor = np.clip(tensor, 0, 1)

    # Transpose from (C, H, W) to (H, W, C)
    tensor = tensor.transpose(1, 2, 0)

    return tensor


def visualize_count_comparison(image_batch, pred_counts, gt_counts, save_path=None, max_images=8):
    """
    Visualize the comparison between predicted and ground truth counts.

    Args:
        image_batch (torch.Tensor): Batch of images with shape (B, C, H, W).
        pred_counts (torch.Tensor): Predicted counts with shape (B,).
        gt_counts (torch.Tensor): Ground truth counts with shape (B,).
        save_path (str, optional): Path to save the visualization.
        max_images (int): Maximum number of images to visualize.

    Returns:
        matplotlib.figure.Figure: Matplotlib figure with the visualization.
    """
    batch_size = min(image_batch.size(0), max_images)

    # Create a grid of images
    fig, axes = plt.subplots(batch_size, 1, figsize=(8, 3 * batch_size))

    # Handle the case of a single image
    if batch_size == 1:
        axes = [axes]

    for i in range(batch_size):
        # Denormalize image
        img = denormalize_image(image_batch[i])

        # Get counts
        pred_count = pred_counts[i].item()
        gt_count = gt_counts[i].item()

        # Display image and counts
        axes[i].imshow(img)
        axes[i].set_title(f'Predicted: {pred_count:.1f}, Ground Truth: {gt_count:.1f}')
        axes[i].axis('off')

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.close(fig)

    return fig


def visualize_density_map(image, points, sigma=15, save_path=None):
    """
    Visualize density map generated from point annotations.

    Args:
        image (torch.Tensor or numpy.ndarray): Image tensor with shape (C, H, W) or numpy array.
        points (numpy.ndarray): Array of points with shape (N, 2).
        sigma (float): Sigma for Gaussian kernel.
        save_path (str, optional): Path to save the visualization.

    Returns:
        matplotlib.figure.Figure: Matplotlib figure with the visualization.
    """
    # Convert to numpy if tensor
    if isinstance(image, torch.Tensor):
        image = denormalize_image(image)

    # Get image dimensions
    h, w = image.shape[:2]

    # Create density map
    density_map = np.zeros((h, w), dtype=np.float32)

    # If there are no points, return empty density map
    if points.shape[0] == 0:
        density_map = np.zeros((h, w), dtype=np.float32)
    else:
        # Create density map by placing Gaussian at each point
        for i in range(points.shape[0]):
            point_x, point_y = points[i]
            point_x, point_y = min(w - 1, max(0, int(point_x))), min(h - 1, max(0, int(point_y)))
            density_map[point_y, point_x] = 1

        # Apply Gaussian filter
        density_map = gaussian_filter(density_map, sigma=sigma, truncate=3.0 / sigma)

        # Normalize density map to preserve count
        if density_map.sum() > 0:
            density_map = density_map / density_map.sum() * points.shape[0]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display image with points
    axes[0].imshow(image)
    axes[0].scatter(points[:, 0], points[:, 1], c='red', s=1)
    axes[0].set_title(f'Image with {points.shape[0]} people')
    axes[0].axis('off')

    # Display density map
    density_plot = axes[1].imshow(density_map, cmap=cm.jet)
    axes[1].set_title(f'Density Map (Sum: {density_map.sum():.1f})')
    axes[1].axis('off')

    # Add colorbar
    cbar = plt.colorbar(density_plot, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Density')

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig


def visualize_training_progress(metrics_history, save_path=None):
    """
    Visualize training progress with metrics history.

    Args:
        metrics_history (dict): Dictionary containing metrics history.
        save_path (str, optional): Path to save the visualization.

    Returns:
        matplotlib.figure.Figure: Matplotlib figure with the visualization.
    """
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot loss curves
    if 'train_loss' in metrics_history and 'val_loss' in metrics_history:
        epochs = range(1, len(metrics_history['train_loss']) + 1)
        axes[0].plot(epochs, metrics_history['train_loss'], 'b-', label='Training Loss')
        axes[0].plot(epochs, metrics_history['val_loss'], 'r-', label='Validation Loss')
        axes[0].set_title('Loss Curves')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

    # Plot MAE curves
    if 'train_mae' in metrics_history and 'val_mae' in metrics_history:
        epochs = range(1, len(metrics_history['train_mae']) + 1)
        axes[1].plot(epochs, metrics_history['train_mae'], 'b-', label='Training MAE')
        axes[1].plot(epochs, metrics_history['val_mae'], 'r-', label='Validation MAE')
        axes[1].set_title('MAE Curves')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig


def visualize_model_predictions(model, dataloader, device, save_dir=None, num_samples=4):
    """
    Visualize model predictions on a set of samples.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for evaluation.
        device (torch.device): Device to run evaluation on.
        save_dir (str, optional): Directory to save visualizations.
        num_samples (int): Number of samples to visualize.

    Returns:
        list: List of matplotlib figures.
    """
    model.eval()
    figures = []

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            images = batch['image'].to(device)
            gt_counts = batch['count']

            # Get predictions
            pred_counts = model(images)

            # Create visualization
            fig = visualize_count_comparison(
                images, pred_counts, gt_counts,
                save_path=os.path.join(save_dir, f'sample_{i}.png') if save_dir else None,
                max_images=1
            )

            figures.append(fig)

    return figures


def visualize_model_performance(metrics_dict, save_path=None):
    """
    Visualize model performance using evaluation metrics.

    Args:
        metrics_dict (dict): Dictionary containing evaluation metrics.
        save_path (str, optional): Path to save the visualization.

    Returns:
        matplotlib.figure.Figure: Matplotlib figure with the visualization.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get metrics
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    # Create bar plot
    bars = ax.bar(metrics, values, color='skyblue')

    # Add labels and values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_title('Model Performance Metrics')
    ax.set_ylabel('Value')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig