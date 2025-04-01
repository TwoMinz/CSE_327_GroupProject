"""
Evaluation metrics for crowd counting models.
"""

import torch
import numpy as np


def mean_absolute_error(pred, target):
    """
    Calculate mean absolute error.

    Args:
        pred (torch.Tensor or numpy.ndarray): Predicted values.
        target (torch.Tensor or numpy.ndarray): Target values.

    Returns:
        float: Mean absolute error.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    return np.mean(np.abs(pred - target))


def mean_squared_error(pred, target):
    """
    Calculate mean squared error.

    Args:
        pred (torch.Tensor or numpy.ndarray): Predicted values.
        target (torch.Tensor or numpy.ndarray): Target values.

    Returns:
        float: Mean squared error.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    return np.mean((pred - target) ** 2)


def root_mean_squared_error(pred, target):
    """
    Calculate root mean squared error.

    Args:
        pred (torch.Tensor or numpy.ndarray): Predicted values.
        target (torch.Tensor or numpy.ndarray): Target values.

    Returns:
        float: Root mean squared error.
    """
    return np.sqrt(mean_squared_error(pred, target))


def mean_absolute_percentage_error(pred, target, epsilon=1e-4):
    """
    Calculate mean absolute percentage error.

    Args:
        pred (torch.Tensor or numpy.ndarray): Predicted values.
        target (torch.Tensor or numpy.ndarray): Target values.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: Mean absolute percentage error.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    return np.mean(np.abs((pred - target) / (target + epsilon))) * 100


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on dataloader.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for evaluation.
        device (torch.device): Device to run evaluation on.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = batch['count'].to(device)

            outputs = model(images)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    mae = mean_absolute_error(all_preds, all_targets)
    mse = mean_squared_error(all_preds, all_targets)
    rmse = root_mean_squared_error(all_preds, all_targets)
    mape = mean_absolute_percentage_error(all_preds, all_targets)

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape
    }