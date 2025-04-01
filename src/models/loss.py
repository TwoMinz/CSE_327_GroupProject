"""
Loss functions for crowd counting models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CountLoss(nn.Module):
    """
    Basic count loss based on Mean Squared Error (MSE).

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Options: 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, reduction='mean'):
        super(CountLoss, self).__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, pred_count, gt_count):
        """
        Args:
            pred_count (torch.Tensor): Predicted count, shape (B,)
            gt_count (torch.Tensor): Ground truth count, shape (B,)

        Returns:
            torch.Tensor: Loss value
        """
        return self.mse_loss(pred_count, gt_count)


class WeightedCountLoss(nn.Module):
    """
    Weighted count loss that gives higher weights to samples with more people.

    Args:
        alpha (float): Weight factor for balancing the loss. Default: 0.1.
        reduction (str): Specifies the reduction to apply to the output.
            Options: 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, alpha=0.1, reduction='mean'):
        super(WeightedCountLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred_count, gt_count):
        """
        Args:
            pred_count (torch.Tensor): Predicted count, shape (B,)
            gt_count (torch.Tensor): Ground truth count, shape (B,)

        Returns:
            torch.Tensor: Loss value
        """
        # Calculate weights based on ground truth counts
        weights = torch.exp(self.alpha * gt_count)
        weights = weights / weights.mean()

        # Calculate squared error
        squared_error = (pred_count - gt_count) ** 2

        # Apply weights
        weighted_squared_error = weights * squared_error

        # Apply reduction
        if self.reduction == 'mean':
            return weighted_squared_error.mean()
        elif self.reduction == 'sum':
            return weighted_squared_error.sum()
        else:  # 'none'
            return weighted_squared_error


class RelativeCountLoss(nn.Module):
    """
    Relative count loss based on Relative Mean Squared Error (RMSE).

    Args:
        epsilon (float): Small constant to avoid division by zero. Default: 1e-4.
        reduction (str): Specifies the reduction to apply to the output.
            Options: 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, epsilon=1e-4, reduction='mean'):
        super(RelativeCountLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred_count, gt_count):
        """
        Args:
            pred_count (torch.Tensor): Predicted count, shape (B,)
            gt_count (torch.Tensor): Ground truth count, shape (B,)

        Returns:
            torch.Tensor: Loss value
        """
        # Calculate relative error
        relative_error = ((pred_count - gt_count) / (gt_count + self.epsilon)) ** 2

        # Apply reduction
        if self.reduction == 'mean':
            return relative_error.mean()
        elif self.reduction == 'sum':
            return relative_error.sum()
        else:  # 'none'
            return relative_error


class CombinedLoss(nn.Module):
    """
    Combined loss function that combines MSE loss with relative loss.

    Args:
        mse_weight (float): Weight for MSE loss. Default: 1.0.
        rel_weight (float): Weight for relative loss. Default: 0.1.
        epsilon (float): Small constant to avoid division by zero. Default: 1e-4.
        reduction (str): Specifies the reduction to apply to the output.
            Options: 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, mse_weight=1.0, rel_weight=0.1, epsilon=1e-4, reduction='mean'):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.rel_weight = rel_weight
        self.reduction = reduction
        self.mse_loss = CountLoss(reduction=reduction)
        self.rel_loss = RelativeCountLoss(epsilon=epsilon, reduction=reduction)

    def forward(self, pred_count, gt_count):
        """
        Args:
            pred_count (torch.Tensor): Predicted count, shape (B,)
            gt_count (torch.Tensor): Ground truth count, shape (B,)

        Returns:
            torch.Tensor: Loss value
        """
        mse_loss = self.mse_loss(pred_count, gt_count)
        rel_loss = self.rel_loss(pred_count, gt_count)

        return self.mse_weight * mse_loss + self.rel_weight * rel_loss