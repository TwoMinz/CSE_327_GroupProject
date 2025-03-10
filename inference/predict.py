import os
import sys
import argparse
import yaml
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.vit_model import CrowdViT, load_model_config, create_model_from_config


def load_model(checkpoint_path, config_path=None):
    """
    Load trained model from checkpoint

    Args:
        checkpoint_path (str): Path to model checkpoint
        config_path (str, optional): Path to model configuration file

    Returns:
        tuple: (model, config)
    """
    # Load configuration
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        # Try to find config in the same directory as checkpoint
        config_in_checkpoint_dir = os.path.join(
            os.path.dirname(checkpoint_path), "config.yaml"
        )
        if os.path.exists(config_in_checkpoint_dir):
            with open(config_in_checkpoint_dir, "r") as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(
                f"No configuration file found. Please provide a valid config_path."
            )

    # Create model
    model, _ = create_model_from_config(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    return model, config


def process_image(image_path, config, device="cuda"):
    """
    Process and prepare an image for inference

    Args:
        image_path (str): Path to the image file
        config (dict): Model configuration
        device (str): Device to use for inference

    Returns:
        torch.Tensor: Processed image tensor
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get target size from config
    target_size = tuple(config["data"]["image_size"])

    # Create transform
    transform = A.Compose(
        [
            A.Resize(height=target_size[0], width=target_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    # Apply transform
    transformed = transform(image=image)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    return image_tensor, image


def predict_wait_time(model, image_tensor, device="cuda"):
    """
    Predict wait time from an image

    Args:
        model (nn.Module): Trained model
        image_tensor (torch.Tensor): Processed image tensor
        device (str): Device to use for inference

    Returns:
        dict: Prediction results
    """
    # Ensure model is in evaluation mode
    model.eval()
    model = model.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)

    # Extract results
    results = {}
    for key, value in outputs.items():
        results[key] = value.item()

    return results


def visualize_prediction(image, results, output_path=None, show=True):
    """
    Visualize prediction results

    Args:
        image (np.ndarray): Original image
        results (dict): Prediction results
        output_path (str, optional): Path to save the visualization
        show (bool): Whether to display the visualization
    """
    # Create figure
    plt.figure(figsize=(12, 8))

    # Show image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    # Show predictions
    plt.subplot(1, 2, 2)
    plt.axis("off")

    # Create text for predictions
    text = "Prediction Results:\n\n"

    if "wait_time" in results:
        text += f"Estimated Wait Time: {results['wait_time']:.1f} minutes\n\n"

    if "people_count" in results:
        text += f"Estimated People Count: {results['people_count']:.1f}\n\n"

    # Add guidelines for wait time
    text += "Wait Time Guidelines:\n"
    text += "- < 5 minutes: Very short wait\n"
    text += "- 5-15 minutes: Short wait\n"
    text += "- 15-30 minutes: Moderate wait\n"
    text += "- 30-45 minutes: Long wait\n"
    text += "- > 45 minutes: Very long wait"

    plt.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=12,
        transform=plt.gca().transAxes,
    )

    plt.tight_layout()

    # Save visualization if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    # Show visualization
    if show:
        plt.show()
    else:
        plt.close()


def batch_inference(model, image_dir, output_dir, config, device="cuda"):
    """
    Run inference on a batch of images

    Args:
        model (nn.Module): Trained model
        image_dir (str): Directory containing images
        output_dir (str): Directory to save visualizations
        config (dict): Model configuration
        device (str): Device to use for inference
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        print(f"No image files found in {image_dir}")
        return

    print(f"Running inference on {len(image_files)} images...")

    # Process each image
    results_summary = []

    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        output_path = os.path.join(
            output_dir, f"{os.path.splitext(image_file)[0]}_prediction.png"
        )

        try:
            # Process image
            image_tensor, image = process_image(image_path, config, device)

            # Make prediction
            results = predict_wait_time(model, image_tensor, device)

            # Visualize prediction
            visualize_prediction(image, results, output_path, show=False)

            # Collect results
            results["image_file"] = image_file
            results_summary.append(results)

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

    # Save summary results
    summary_path = os.path.join(output_dir, "inference_summary.csv")

    if results_summary:
        import pandas as pd

        # Convert to DataFrame
        df = pd.DataFrame(results_summary)

        # Reorder columns to put image_file first
        cols = ["image_file"] + [col for col in df.columns if col != "image_file"]
        df = df[cols]

        # Save to CSV
        df.to_csv(summary_path, index=False)

        print(f"Results saved to {summary_path}")

    print(f"Visualizations saved to {output_dir}")


def process_video(
    model, video_path, output_path, config, device="cuda", frame_interval=30
):
    """
    Process a video and add wait time predictions

    Args:
        model (nn.Module): Trained model
        video_path (str): Path to input video
        output_path (str): Path to output video
        config (dict): Model configuration
        device (str): Device to use for inference
        frame_interval (int): Interval between processed frames
    """
    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video at {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Get target size from config
    target_size = tuple(config["data"]["image_size"])

    # Create transform
    transform = A.Compose(
        [
            A.Resize(height=target_size[0], width=target_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    # Process video
    frame_count = 0
    last_prediction = None

    try:
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Process frame every `frame_interval` frames or if it's the first frame
                if frame_count % frame_interval == 0 or frame_count == 0:
                    # Convert frame from BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Apply transform
                    transformed = transform(image=rgb_frame)
                    image_tensor = transformed["image"].unsqueeze(0).to(device)

                    # Make prediction
                    last_prediction = predict_wait_time(model, image_tensor, device)

                # Add prediction text to frame
                if last_prediction:
                    # Create text
                    wait_time_text = (
                        f"Wait Time: {last_prediction['wait_time']:.1f} min"
                    )

                    if "people_count" in last_prediction:
                        people_count_text = (
                            f"People Count: {last_prediction['people_count']:.1f}"
                        )
                    else:
                        people_count_text = ""

                    # Add background rectangle for better readability
                    cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)

                    # Add text
                    cv2.putText(
                        frame,
                        wait_time_text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    if people_count_text:
                        cv2.putText(
                            frame,
                            people_count_text,
                            (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )

                # Write frame
                out.write(frame)

                # Update counters
                frame_count += 1
                pbar.update(1)

    finally:
        # Release resources
        cap.release()
        out.release()

    print(f"Processed video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with CrowdViT model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to model configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["image", "batch", "video"],
        default="image",
        help="Inference mode",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image, directory, or video",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/inference",
        help="Path to output directory or file",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for inference"
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=30,
        help="Frame interval for video processing",
    )
    args = parser.parse_args()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead")
        args.device = "cpu"

    # Load model
    model, config = load_model(args.checkpoint, args.config)
    model.to(args.device)

    # Run inference based on mode
    if args.mode == "image":
        # Single image inference
        image_tensor, image = process_image(args.input, config, args.device)
        results = predict_wait_time(model, image_tensor, args.device)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        # Visualize prediction
        visualize_prediction(image, results, args.output)

        print(f"Prediction results:")
        for key, value in results.items():
            print(f"  {key}: {value:.2f}")

    elif args.mode == "batch":
        # Batch inference
        batch_inference(model, args.input, args.output, config, args.device)

    elif args.mode == "video":
        # Video processing
        process_video(
            model, args.input, args.output, config, args.device, args.frame_interval
        )


if __name__ == "__main__":
    main()
