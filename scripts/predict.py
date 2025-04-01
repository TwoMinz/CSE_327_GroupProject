"""
Prediction script for TransCrowd model.
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.config import MODEL_CONFIG, DEVICE, CHECKPOINT_DIR, OUTPUT_DIR
from src.models.transcrowd import TransCrowd
from src.data_utils.dataset import get_transforms


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


def predict_image(model, image_path, device, transform=None):
    """
    Predict crowd count for an image.

    Args:
        model (nn.Module): Model to use for prediction.
        image_path (str): Path to image.
        device (torch.device): Device to run prediction on.
        transform (callable, optional): Transform to apply to image.

    Returns:
        float: Predicted crowd count.
    """
    # Load image
    img = Image.open(image_path).convert('RGB')

    # Apply transform
    if transform is None:
        _, transform = get_transforms(MODEL_CONFIG['img_size'])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        pred_count = model(img_tensor).item()

    return pred_count


def visualize_prediction(image_path, pred_count, save_path=None):
    """
    Visualize prediction on image.

    Args:
        image_path (str): Path to image.
        pred_count (float): Predicted crowd count.
        save_path (str, optional): Path to save visualization.

    Returns:
        matplotlib.figure.Figure: Matplotlib figure with visualization.
    """
    # Load image
    img = Image.open(image_path).convert('RGB')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Display image
    ax.imshow(img)
    ax.set_title(f'Predicted Count: {pred_count:.1f}')
    ax.axis('off')

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig


def predict(args):
    """
    Predict crowd count for images.

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

    # Create output directory
    output_dir = os.path.join(OUTPUT_DIR, 'predictions')
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, DEVICE)

    # Get transform
    _, transform = get_transforms(MODEL_CONFIG['img_size'])

    # Process images
    if os.path.isdir(args.input):
        # Process all images in directory
        image_paths = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    else:
        # Process single image
        image_paths = [args.input]

    # Predict for each image
    results = []
    for image_path in image_paths:
        print(f"Processing {image_path}...")

        # Predict
        pred_count = predict_image(model, image_path, DEVICE, transform)

        # Visualize
        output_filename = os.path.basename(image_path).split('.')[0] + '_prediction.png'
        output_path = os.path.join(output_dir, output_filename)
        visualize_prediction(image_path, pred_count, output_path)

        # Save result
        results.append({
            'image_path': image_path,
            'predicted_count': pred_count
        })

        print(f"  Predicted count: {pred_count:.1f}")
        print(f"  Visualization saved to {output_path}")

    # Print summary
    print("\nPrediction Summary:")
    for result in results:
        print(f"  {os.path.basename(result['image_path'])}: {result['predicted_count']:.1f}")

    print(f"\nAll predictions completed. Results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict with TransCrowd model')
    parser.add_argument('--input', type=str, required=True, help='path to input image or directory')
    parser.add_argument('--name', type=str, default=None, help='experiment name')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint')
    args = parser.parse_args()

    predict(args)