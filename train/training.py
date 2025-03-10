import os
import sys
import time
import yaml
import json
import argparse
from tqdm import tqdm
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.vit_model import (
    CrowdViT,
    CrowdDensityLoss,
    load_model_config,
    create_model_from_config,
)
from data.dataset import ShanghaiTechDataset, create_dataloaders
from utils.metrics import compute_metrics, MetricTracker


def check_gpu_availability():
    """Check if GPU is available and print GPU info"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA device(s):")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (
                1024**3
            )  # Convert to GB
            print(f"  Device {i}: {device_name} with {total_memory:.2f} GB memory")
        return True
    else:
        print("No GPU available, using CPU")
        return False


def train_one_epoch(
    model, train_loader, criterion, optimizer, device, epoch, metric_tracker
):
    """Train the model for one epoch"""
    model.train()
    metric_tracker.reset()

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")

    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        images = batch["image"].to(device)
        wait_time = batch["wait_time"].to(device)
        num_people = batch["num_people"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        targets = {"wait_time": wait_time, "num_people": num_people}
        loss, loss_dict = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update metrics
        batch_size = images.size(0)
        metric_tracker.update(loss_dict, outputs, targets, batch_size)

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{metric_tracker.avg_metrics['total_loss']:.4f}",
                "wait_mae": f"{metric_tracker.avg_metrics['wait_time_mae']:.2f}",
            }
        )

    return metric_tracker.get_metrics()


def validate(model, val_loader, criterion, device, metric_tracker):
    """Validate the model"""
    model.eval()
    metric_tracker.reset()

    progress_bar = tqdm(val_loader, desc="Validation")

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch["image"].to(device)
            wait_time = batch["wait_time"].to(device)
            num_people = batch["num_people"].to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            targets = {"wait_time": wait_time, "num_people": num_people}
            loss, loss_dict = criterion(outputs, targets)

            # Update metrics
            batch_size = images.size(0)
            metric_tracker.update(loss_dict, outputs, targets, batch_size)

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{metric_tracker.avg_metrics['total_loss']:.4f}",
                    "wait_mae": f"{metric_tracker.avg_metrics['wait_time_mae']:.2f}",
                }
            )

    return metric_tracker.get_metrics()


def save_checkpoint(
    model, optimizer, scheduler, epoch, val_metrics, checkpoint_dir, is_best=False
):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "val_metrics": val_metrics,
    }

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    torch.save(checkpoint, latest_path)

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        epoch_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(checkpoint, epoch_path)

    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(checkpoint, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at '{checkpoint_path}'")
        return 0, {}

    print(f"Loading checkpoint from '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if (
        scheduler is not None
        and "scheduler" in checkpoint
        and checkpoint["scheduler"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = checkpoint.get("epoch", 0)
    val_metrics = checkpoint.get("val_metrics", {})

    return epoch, val_metrics


def create_scheduler(optimizer, config, num_training_steps):
    """Create learning rate scheduler"""
    warmup_epochs = config["training"].get("warmup_epochs", 0)
    num_epochs = config["training"]["num_epochs"]

    if warmup_epochs > 0:
        # Warmup phase with linear learning rate increase
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs * num_training_steps,
        )

        # Cosine annealing phase
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=(num_epochs - warmup_epochs) * num_training_steps,
            eta_min=1e-6,
        )

        # Combine schedulers
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[(warmup_epochs * num_training_steps)],
        )
    else:
        # Only cosine annealing
        scheduler = CosineAnnealingLR(
            optimizer, T_max=num_epochs * num_training_steps, eta_min=1e-6
        )

    return scheduler


def train(config, resume_from=None):
    """Main training function"""
    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Check GPU availability
    use_gpu = check_gpu_availability()
    device = torch.device("cuda" if use_gpu else "cpu")

    # Create model, criterion, and optimizer
    model, criterion = create_model_from_config(config)
    model = model.to(device)

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        root_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=4,  # Adjust based on available CPU cores
        part=config["data"]["part"],
        target_size=tuple(config["data"]["image_size"]),
    )

    # Create scheduler
    steps_per_epoch = len(train_loader)
    scheduler = create_scheduler(optimizer, config, steps_per_epoch)

    # Create metric tracker
    metric_tracker = MetricTracker()

    # Setup TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(
        "./outputs/logs", f"{config['model']['vit_model']}_{timestamp}"
    )
    writer = SummaryWriter(log_dir=log_dir)

    # Setup checkpointing
    checkpoint_dir = os.path.join(
        "./outputs/checkpoints", f"{config['model']['vit_model']}_{timestamp}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Resume from checkpoint if provided
    start_epoch = 0
    best_wait_time_mae = float("inf")

    if resume_from:
        start_epoch, val_metrics = load_checkpoint(
            resume_from, model, optimizer, scheduler
        )
        if val_metrics and "wait_time_mae" in val_metrics:
            best_wait_time_mae = val_metrics["wait_time_mae"]
        print(
            f"Resuming from epoch {start_epoch} with best wait time MAE: {best_wait_time_mae:.2f}"
        )

    # Training loop
    num_epochs = config["training"]["num_epochs"]
    early_stopping_patience = config["training"].get(
        "early_stopping_patience", num_epochs
    )
    no_improvement_count = 0

    print(f"Starting training for {num_epochs} epochs")
    print(f"Training device: {device}")
    print(f"Dataset: {config['data']['dataset']} Part {config['data']['part']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Image size: {config['data']['image_size']}")
    print(f"Model: {config['model']['vit_model']}")

    for epoch in range(start_epoch, num_epochs):
        # Train for one epoch
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, metric_tracker
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, metric_tracker)

        # Log metrics
        for metric_name, metric_value in train_metrics.items():
            writer.add_scalar(f"train/{metric_name}", metric_value, epoch)

        for metric_name, metric_value in val_metrics.items():
            writer.add_scalar(f"val/{metric_name}", metric_value, epoch)

        # Check if current model is best
        current_wait_time_mae = val_metrics["wait_time_mae"]
        is_best = current_wait_time_mae < best_wait_time_mae

        if is_best:
            best_wait_time_mae = current_wait_time_mae
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_metrics, checkpoint_dir, is_best
        )

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{num_epochs} summary:")
        print(
            f"  Train loss: {train_metrics['total_loss']:.4f}, Wait time MAE: {train_metrics['wait_time_mae']:.2f}min"
        )
        print(
            f"  Val loss: {val_metrics['total_loss']:.4f}, Wait time MAE: {val_metrics['wait_time_mae']:.2f}min"
        )

        if "people_count_mae" in train_metrics:
            print(f"  Train people count MAE: {train_metrics['people_count_mae']:.2f}")
            print(f"  Val people count MAE: {val_metrics['people_count_mae']:.2f}")

        print(f"  Best val wait time MAE: {best_wait_time_mae:.2f}min")

        # Early stopping
        if no_improvement_count >= early_stopping_patience:
            print(
                f"No improvement for {early_stopping_patience} epochs, stopping training"
            )
            break

    writer.close()
    print("Training completed!")
    print(f"Best validation wait time MAE: {best_wait_time_mae:.2f} minutes")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")

    return best_wait_time_mae, checkpoint_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CrowdViT model")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/model_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Start training
    best_mae, checkpoint_dir = train(config, args.resume)
