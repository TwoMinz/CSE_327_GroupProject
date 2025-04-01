"""
Training script for TransCrowd model.
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.config import MODEL_CONFIG, TRAIN_CONFIG, DEVICE, CHECKPOINT_DIR, LOG_DIR, get_experiment_name
from src.data_utils.dataset import get_dataloaders
from src.models.transcrowd import TransCrowd, ensure_model_on_device
from src.models.loss import CombinedLoss
from src.utils.metrics import evaluate_model, mean_absolute_error
from src.utils.visualization import visualize_training_progress, visualize_model_predictions


def set_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(model, config):
    """
    Get optimizer based on config.

    Args:
        model (nn.Module): Model.
        config (dict): Training configuration.

    Returns:
        torch.optim.Optimizer: Optimizer.
    """
    return optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )


def get_scheduler(optimizer, config):
    """
    Get learning rate scheduler based on config.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer.
        config (dict): Training configuration.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler.
    """
    if config['lr_scheduler'] == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'] - config['warmup_epochs'],
            eta_min=1e-6
        )
    elif config['lr_scheduler'] == 'step':
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config['lr_decay_epochs'],
            gamma=config['lr_decay_rate']
        )
    else:
        return None


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
    """
    Train model for one epoch.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training dataloader.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run training on.
        epoch (int): Current epoch.
        config (dict): Training configuration.

    Returns:
        float: Average training loss.
        float: Average MAE.
    """
    model.train()
    total_loss = 0
    total_mae = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["num_epochs"]}')
    for i, batch in enumerate(pbar):
        images = batch['image'].to(device)
        counts = batch['count'].to(device)

        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, counts)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config['grad_clip_norm'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])

        optimizer.step()

        # Calculate metrics
        total_loss += loss.item()
        mae = mean_absolute_error(predictions.detach(), counts.detach())
        total_mae += mae

        # Update progress bar
        if i % config['log_freq'] == 0:
            pbar.set_postfix({
                'loss': loss.item(),
                'mae': mae
            })

    # Calculate average metrics
    avg_loss = total_loss / len(train_loader)
    avg_mae = total_mae / len(train_loader)

    return avg_loss, avg_mae


def validate(model, val_loader, criterion, device):
    """
    Validate model on validation set.

    Args:
        model (nn.Module): Model to validate.
        val_loader (DataLoader): Validation dataloader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run validation on.

    Returns:
        float: Average validation loss.
        float: Average MAE.
    """
    model.eval()
    total_loss = 0
    total_mae = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            counts = batch['count'].to(device)

            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, counts)

            # Calculate metrics
            total_loss += loss.item()
            mae = mean_absolute_error(predictions, counts)
            total_mae += mae

    # Calculate average metrics
    avg_loss = total_loss / len(val_loader)
    avg_mae = total_mae / len(val_loader)

    return avg_loss, avg_mae


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, checkpoint_dir, is_best=False):
    """
    Save model checkpoint.

    Args:
        model (nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        epoch (int): Current epoch.
        metrics (dict): Training metrics.
        config (dict): Training configuration.
        checkpoint_dir (str): Directory to save checkpoint.
        is_best (bool): Whether this is the best model so far.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Save latest checkpoint
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest.pth'))

    # Save epoch checkpoint
    if (epoch + 1) % config['save_freq'] == 0:
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'epoch_{epoch + 1}.pth'))

    # Save best checkpoint
    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'best.pth'))


def train(args):
    """
    Main training function.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Set random seed
    set_seed(TRAIN_CONFIG['seed'])

    # Set up experiment
    experiment_name = get_experiment_name() if args.name is None else args.name
    experiment_dir = os.path.join(LOG_DIR, experiment_name)
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, experiment_name)

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        part=TRAIN_CONFIG['part'],
        batch_size=TRAIN_CONFIG['batch_size'],
        num_workers=TRAIN_CONFIG['num_workers'],
        pin_memory=TRAIN_CONFIG['pin_memory'],
        img_size=MODEL_CONFIG['img_size']
    )

    # Create model
    model = TransCrowd(MODEL_CONFIG)
    model = ensure_model_on_device(model, DEVICE)
    model = model.to(DEVICE)

    # Create criterion, optimizer, and scheduler
    criterion = CombinedLoss(mse_weight=1.0, rel_weight=0.1)
    optimizer = get_optimizer(model, TRAIN_CONFIG)
    scheduler = get_scheduler(optimizer, TRAIN_CONFIG)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_mae = float('inf')
    metrics_history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}

    if args.resume:
        checkpoint_path = args.resume
        if os.path.isfile(checkpoint_path):
            print(f"=> loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if 'metrics' in checkpoint:
                best_val_mae = checkpoint['metrics'].get('best_val_mae', float('inf'))
                metrics_history = checkpoint['metrics'].get('history', metrics_history)

            print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{checkpoint_path}'")

    # Training loop
    for epoch in range(start_epoch, TRAIN_CONFIG['num_epochs']):
        # Train for one epoch
        train_loss, train_mae = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch, TRAIN_CONFIG
        )

        # Validate
        if (epoch + 1) % TRAIN_CONFIG['val_freq'] == 0:
            val_loss, val_mae = validate(model, val_loader, criterion, DEVICE)

            # Update learning rate
            if scheduler is not None:
                scheduler.step()

            # Print validation results
            print(f"Epoch {epoch + 1}/{TRAIN_CONFIG['num_epochs']} - "
                  f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f}, "
                  f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}")

            # Update metrics history
            metrics_history['train_loss'].append(train_loss)
            metrics_history['val_loss'].append(val_loss)
            metrics_history['train_mae'].append(train_mae)
            metrics_history['val_mae'].append(val_mae)

            # Save checkpoint
            is_best = val_mae < best_val_mae
            if is_best:
                best_val_mae = val_mae

            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'best_val_mae': best_val_mae, 'history': metrics_history},
                TRAIN_CONFIG, checkpoint_dir, is_best
            )

            # Visualize training progress
            visualize_training_progress(
                metrics_history,
                save_path=os.path.join(experiment_dir, 'training_progress.png')
            )

            # Visualize model predictions
            visualize_model_predictions(
                model, val_loader, DEVICE,
                save_dir=os.path.join(experiment_dir, f'epoch_{epoch + 1}_predictions'),
                num_samples=2
            )

            # Early stopping
            if len(metrics_history['val_mae']) >= TRAIN_CONFIG['early_stopping_patience']:
                if all(metrics_history['val_mae'][-i - 1] <= metrics_history['val_mae'][-i]
                       for i in range(1, TRAIN_CONFIG['early_stopping_patience'])):
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

    # Final evaluation
    print("Training completed. Final evaluation...")
    metrics = evaluate_model(model, val_loader, DEVICE)
    print(f"Final evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TransCrowd model')
    parser.add_argument('--name', type=str, default=None, help='experiment name')
    parser.add_argument('--resume', type=str, default=None, help='path to latest checkpoint')
    args = parser.parse_args()

    train(args)
