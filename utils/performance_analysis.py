import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_data(log_dir):
    """
    Load training metrics from TensorBoard logs.

    Args:
        log_dir (str): Path to TensorBoard log directory

    Returns:
        dict: Dictionary of metric names to pandas dataframes
    """
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Get available scalars
    tags = event_acc.Tags()["scalars"]

    data = {}
    for tag in tags:
        events = event_acc.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        data[tag] = pd.DataFrame({"step": steps, "value": values})

    return data


def compare_metrics(cuda_data, mps_data, output_dir):
    """
    Compare metrics between CUDA and MPS runs.

    Args:
        cuda_data (dict): Data from CUDA run
        mps_data (dict): Data from MPS run
        output_dir (str): Directory to save comparison plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find common metrics between both runs
    common_metrics = set(cuda_data.keys()).intersection(set(mps_data.keys()))

    for metric in common_metrics:
        plt.figure(figsize=(10, 6))

        # Plot CUDA data
        plt.plot(
            cuda_data[metric]["step"],
            cuda_data[metric]["value"],
            "b-",
            label="CUDA (Windows)",
        )

        # Plot MPS data
        plt.plot(
            mps_data[metric]["step"], mps_data[metric]["value"], "r-", label="MPS (Mac)"
        )

        plt.title(f"Comparison of {metric}")
        plt.xlabel("Training Step")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()

        # Save the figure
        plt.savefig(
            os.path.join(output_dir, f"{metric.replace('/', '_')}_comparison.png")
        )
        plt.close()


def compare_training_speed(cuda_data, mps_data):
    """
    Compare training speed between CUDA and MPS.

    Args:
        cuda_data (dict): Data from CUDA run
        mps_data (dict): Data from MPS run

    Returns:
        dict: Dictionary with speed comparison statistics
    """
    # Find a common metric to use for step timing
    # (e.g., 'train/total_loss' should exist in both)
    common_train_metrics = [m for m in cuda_data.keys() if "train/" in m]

    if not common_train_metrics:
        return {"error": "No common training metrics found"}

    metric = common_train_metrics[0]

    # Get the maximum step for both
    cuda_max_step = cuda_data[metric]["step"].max()
    mps_max_step = mps_data[metric]["step"].max()

    # Assuming steps are evenly spaced, we can use the number of steps as a proxy for speed
    common_steps = min(cuda_max_step, mps_max_step)

    # Compute statistics
    cuda_values = cuda_data[metric][cuda_data[metric]["step"] <= common_steps]["value"]
    mps_values = mps_data[metric][mps_data[metric]["step"] <= common_steps]["value"]

    stats = {
        "common_steps": common_steps,
        "cuda_avg_loss": cuda_values.mean(),
        "mps_avg_loss": mps_values.mean(),
        "loss_ratio": mps_values.mean() / cuda_values.mean()
        if cuda_values.mean() != 0
        else None,
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compare CUDA and MPS training performance"
    )
    parser.add_argument(
        "--cuda-log",
        type=str,
        required=True,
        help="Path to CUDA TensorBoard log directory",
    )
    parser.add_argument(
        "--mps-log",
        type=str,
        required=True,
        help="Path to MPS TensorBoard log directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/performance_comparison",
        help="Directory to save comparison plots",
    )
    args = parser.parse_args()

    print(f"Loading CUDA data from {args.cuda_log}...")
    cuda_data = load_tensorboard_data(args.cuda_log)

    print(f"Loading MPS data from {args.mps_log}...")
    mps_data = load_tensorboard_data(args.mps_log)

    print("Comparing metrics...")
    compare_metrics(cuda_data, mps_data, args.output_dir)

    print("Analyzing training speed...")
    speed_stats = compare_training_speed(cuda_data, mps_data)

    print("\n===== Performance Comparison =====")
    for key, value in speed_stats.items():
        print(f"{key}: {value}")

    print(f"\nComparison plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
