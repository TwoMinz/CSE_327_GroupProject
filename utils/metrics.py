import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Any


def compute_metrics(
    outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute evaluation metrics for model predictions

    Args:
        outputs (Dict[str, torch.Tensor]): Model outputs containing 'wait_time' and optionally 'people_count'
        targets (Dict[str, torch.Tensor]): Ground truth values containing 'wait_time' and 'num_people'

    Returns:
        Dict[str, float]: Dictionary of computed metrics
    """
    metrics = {}

    # Compute metrics for wait time prediction
    if "wait_time" in outputs and "wait_time" in targets:
        wait_time_pred = outputs["wait_time"].detach().cpu()
        wait_time_true = targets["wait_time"].detach().cpu()

        # Mean Absolute Error (MAE)
        wait_time_mae = torch.abs(wait_time_pred - wait_time_true).mean().item()
        metrics["wait_time_mae"] = wait_time_mae

        # Mean Squared Error (MSE)
        wait_time_mse = ((wait_time_pred - wait_time_true) ** 2).mean().item()
        metrics["wait_time_mse"] = wait_time_mse

        # Root Mean Squared Error (RMSE)
        wait_time_rmse = torch.sqrt(torch.tensor(wait_time_mse)).item()
        metrics["wait_time_rmse"] = wait_time_rmse

        # Mean Absolute Percentage Error (MAPE)
        # Avoid division by zero
        idx = wait_time_true != 0
        if idx.sum() > 0:
            wait_time_mape = (
                torch.abs(
                    (wait_time_true[idx] - wait_time_pred[idx]) / wait_time_true[idx]
                )
                .mean()
                .item()
                * 100
            )
            metrics["wait_time_mape"] = wait_time_mape

    # Compute metrics for people count prediction (if available)
    if "people_count" in outputs and "num_people" in targets:
        people_count_pred = outputs["people_count"].detach().cpu()
        people_count_true = targets["num_people"].detach().cpu()

        # Mean Absolute Error (MAE)
        people_count_mae = (
            torch.abs(people_count_pred - people_count_true).mean().item()
        )
        metrics["people_count_mae"] = people_count_mae

        # Mean Squared Error (MSE)
        people_count_mse = ((people_count_pred - people_count_true) ** 2).mean().item()
        metrics["people_count_mse"] = people_count_mse

        # Root Mean Squared Error (RMSE)
        people_count_rmse = torch.sqrt(torch.tensor(people_count_mse)).item()
        metrics["people_count_rmse"] = people_count_rmse

        # Mean Absolute Percentage Error (MAPE)
        # Avoid division by zero
        idx = people_count_true != 0
        if idx.sum() > 0:
            people_count_mape = (
                torch.abs(
                    (people_count_true[idx] - people_count_pred[idx])
                    / people_count_true[idx]
                )
                .mean()
                .item()
                * 100
            )
            metrics["people_count_mape"] = people_count_mape

    return metrics


class MetricTracker:
    """
    Class to track and accumulate metrics during training and validation
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.metrics_sum = {}
        self.count = 0
        self.avg_metrics = {}

    def update(
        self,
        loss_dict: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        batch_size: int,
    ):
        """
        Update metrics with batch results

        Args:
            loss_dict (Dict[str, torch.Tensor]): Dictionary of loss values
            outputs (Dict[str, torch.Tensor]): Model outputs
            targets (Dict[str, torch.Tensor]): Ground truth values
            batch_size (int): Batch size
        """
        # Update count
        self.count += batch_size

        # Update loss metrics
        for loss_name, loss_value in loss_dict.items():
            loss_value = (
                loss_value.item()
                if isinstance(loss_value, torch.Tensor)
                else loss_value
            )
            if loss_name not in self.metrics_sum:
                self.metrics_sum[loss_name] = 0.0
            self.metrics_sum[loss_name] += loss_value * batch_size

        # Compute and update other metrics
        batch_metrics = compute_metrics(outputs, targets)
        for metric_name, metric_value in batch_metrics.items():
            if metric_name not in self.metrics_sum:
                self.metrics_sum[metric_name] = 0.0
            self.metrics_sum[metric_name] += metric_value * batch_size

        # Update average metrics
        self._compute_avg_metrics()

    def _compute_avg_metrics(self):
        """Compute average metrics"""
        if self.count > 0:
            self.avg_metrics = {
                key: value / self.count for key, value in self.metrics_sum.items()
            }
        else:
            self.avg_metrics = {}

    def get_metrics(self) -> Dict[str, float]:
        """
        Get current average metrics

        Returns:
            Dict[str, float]: Dictionary of average metric values
        """
        return self.avg_metrics


if __name__ == "__main__":
    # Test the metrics module

    # Create random data
    batch_size = 8

    # Model outputs
    outputs = {
        "wait_time": torch.randn(batch_size) * 5
        + 15,  # Random wait times centered around 15 minutes
        "people_count": torch.abs(
            torch.randn(batch_size) * 10 + 30
        ),  # Random people counts centered around 30
    }

    # Ground truth
    targets = {
        "wait_time": torch.randn(batch_size) * 5 + 15,  # Random wait times
        "num_people": torch.abs(
            torch.randn(batch_size) * 10 + 30
        ),  # Random people counts
    }

    # Compute metrics
    metrics = compute_metrics(outputs, targets)
    print("Computed metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    # Test MetricTracker
    tracker = MetricTracker()

    # Create loss dictionary
    loss_dict = {
        "wait_time_loss": torch.tensor(0.5),
        "people_count_loss": torch.tensor(0.3),
        "total_loss": torch.tensor(0.8),
    }

    # Update tracker
    tracker.update(loss_dict, outputs, targets, batch_size)

    # Get and print metrics
    tracked_metrics = tracker.get_metrics()
    print("\nTracked metrics:")
    for metric_name, metric_value in tracked_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
