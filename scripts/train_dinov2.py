"""
DINOv2 백본을 위한 개선된 Two-stage 훈련 스크립트
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

from src.config import DEVICE, CHECKPOINT_DIR, LOG_DIR, get_experiment_name
from src.data_utils.dataset import ShanghaiTechDataset, custom_collate_fn
from src.models.transcrowd import TransCrowd
from src.utils.metrics import mean_absolute_error
from src.utils.visualization import visualize_training_progress


# DINOv2 최적화 설정
DINOV2_MODEL_CONFIG = {
    'backbone': 'dinov2',
    'img_size': 392,                        # 14의 배수 (DINOv2 패치 크기)
    'patch_size': 14,
    'in_channels': 3,
    'dinov2_size': 'base',                  # 'small', 'base', 'large' 중 선택
    'pretrained': True,
    'freeze_backbone': True,                # Stage 1에서 freeze
    'dropout_rate': 0.1,
}

DINOV2_TRAIN_CONFIG = {
    'part': 'A',
    'batch_size': 16,
    'num_epochs': 150,

    # Two-stage training
    'stage1_epochs': 30,                    # Stage 1: backbone frozen
    'stage2_epochs': 120,                   # Stage 2: full fine-tuning

    # Stage 1 설정
    'stage1_learning_rate': 1e-3,
    'stage1_weight_decay': 1e-4,

    # Stage 2 설정
    'stage2_learning_rate': 1e-5,
    'stage2_weight_decay': 1e-4,

    'lr_scheduler': 'cosine',
    'warmup_epochs': 5,
    'early_stopping_patience': 20,
    'grad_clip_norm': 0.5,
    'seed': 42,
    'num_workers': 4,
    'pin_memory': True,
    'save_freq': 5,
    'log_freq': 10,
    'val_freq': 1,
}


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dinov2_transforms(img_size=392):
    """DINOv2 최적화 transforms"""
    import torchvision.transforms as transforms

    dinov2_mean = [0.485, 0.456, 0.406]
    dinov2_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=dinov2_mean, std=dinov2_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dinov2_mean, std=dinov2_std)
    ])

    return train_transform, test_transform


def get_dataloaders(config):
    """Get optimized dataloaders for DINOv2"""
    train_transform, test_transform = get_dinov2_transforms(config['img_size'])

    train_dataset = ShanghaiTechDataset(
        part=config['part'],
        split='train',
        transform=train_transform
    )
    test_dataset = ShanghaiTechDataset(
        part=config['part'],
        split='test',
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        collate_fn=custom_collate_fn
    )

    return train_loader, test_loader


class ImprovedCountLoss(nn.Module):
    """Improved loss for DINOv2"""
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_count, gt_count):
        mse_loss = nn.functional.mse_loss(pred_count, gt_count)
        relative_error = torch.mean(torch.abs(pred_count - gt_count) / (gt_count + 1.0))
        return self.alpha * mse_loss + self.beta * relative_error


def get_optimizer_and_scheduler(model, config, stage='stage1'):
    """Get optimizer and scheduler for each stage"""
    if stage == 'stage1':
        lr = config['stage1_learning_rate']
        weight_decay = config['stage1_weight_decay']

        # Freeze backbone
        if hasattr(model, 'model') and hasattr(model.model, 'backbone'):
            for param in model.model.backbone.parameters():
                param.requires_grad = False
            trainable_params = model.model.regression_head.parameters()
        else:
            trainable_params = [p for p in model.parameters() if p.requires_grad]

    else:  # stage2
        lr = config['stage2_learning_rate']
        weight_decay = config['stage2_weight_decay']

        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True

        # Different learning rates
        if hasattr(model, 'model'):
            trainable_params = [
                {'params': model.model.backbone.parameters(), 'lr': lr * 0.1},
                {'params': model.model.regression_head.parameters(), 'lr': lr}
            ]
        else:
            trainable_params = model.parameters()

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    if config['lr_scheduler'] == 'cosine':
        total_epochs = config['stage1_epochs'] if stage == 'stage1' else config['stage2_epochs']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - config['warmup_epochs'],
            eta_min=lr * 0.01
        )
    else:
        scheduler = None

    return optimizer, scheduler


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config, stage):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    total_mae = 0

    pbar = tqdm(train_loader, desc=f'{stage} Epoch {epoch + 1}')
    for i, batch in enumerate(pbar):
        images = batch['image'].to(device)
        counts = batch['count'].to(device)

        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, counts)

        # Backward pass
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
                'loss': f'{loss.item():.2f}',
                'mae': f'{mae:.2f}'
            })

    avg_loss = total_loss / len(train_loader)
    avg_mae = total_mae / len(train_loader)

    return avg_loss, avg_mae


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_mae = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            counts = batch['count'].to(device)

            predictions = model(images)
            loss = criterion(predictions, counts)

            total_loss += loss.item()
            mae = mean_absolute_error(predictions, counts)
            total_mae += mae

    avg_loss = total_loss / len(val_loader)
    avg_mae = total_mae / len(val_loader)

    return avg_loss, avg_mae


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, checkpoint_dir, stage, is_best=False):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'stage': stage,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Save latest checkpoint
    torch.save(checkpoint, os.path.join(checkpoint_dir, f'latest_{stage}.pth'))

    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'best_{stage}.pth'))


def print_model_info(model):
    """Print model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")


def train_dinov2(args):
    """Main training function for DINOv2"""
    # Set seed
    set_seed(DINOV2_TRAIN_CONFIG['seed'])

    # Setup directories
    experiment_name = f"dinov2_{DINOV2_MODEL_CONFIG['dinov2_size']}" if args.name is None else args.name
    experiment_dir = os.path.join(LOG_DIR, experiment_name)
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, experiment_name)

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(DINOV2_TRAIN_CONFIG)

    # Create model
    model = TransCrowd(DINOV2_MODEL_CONFIG).to(DEVICE)
    print_model_info(model)

    # Loss function
    criterion = ImprovedCountLoss(alpha=1.0, beta=0.1)

    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    best_val_mae = float('inf')

    print("\n" + "="*80)
    print("STAGE 1: Training Regression Head (Backbone Frozen)")
    print("="*80)

    # Stage 1: Train regression head only
    model_config = DINOV2_MODEL_CONFIG.copy()
    model_config['freeze_backbone'] = True

    optimizer1, scheduler1 = get_optimizer_and_scheduler(model, DINOV2_TRAIN_CONFIG, 'stage1')
    print_model_info(model)

    for epoch in range(DINOV2_TRAIN_CONFIG['stage1_epochs']):
        # Train
        train_loss, train_mae = train_one_epoch(
            model, train_loader, criterion, optimizer1, DEVICE, epoch, DINOV2_TRAIN_CONFIG, 'Stage1'
        )

        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, DEVICE)

        if scheduler1:
            scheduler1.step()

        # Log results
        print(f"Stage1 Epoch {epoch + 1}/{DINOV2_TRAIN_CONFIG['stage1_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f}, "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}")

        # Save checkpoint
        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae

        save_checkpoint(
            model, optimizer1, scheduler1, epoch,
            {'best_val_mae': best_val_mae}, DINOV2_TRAIN_CONFIG,
            checkpoint_dir, 'stage1', is_best
        )

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

    print(f"\nStage 1 completed. Best validation MAE: {best_val_mae:.2f}")

    print("\n" + "="*80)
    print("STAGE 2: Fine-tuning Full Model")
    print("="*80)

    # Stage 2: Fine-tune full model
    optimizer2, scheduler2 = get_optimizer_and_scheduler(model, DINOV2_TRAIN_CONFIG, 'stage2')
    print_model_info(model)

    stage2_best_mae = best_val_mae

    for epoch in range(DINOV2_TRAIN_CONFIG['stage2_epochs']):
        # Train
        train_loss, train_mae = train_one_epoch(
            model, train_loader, criterion, optimizer2, DEVICE, epoch, DINOV2_TRAIN_CONFIG, 'Stage2'
        )

        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, DEVICE)

        if scheduler2:
            scheduler2.step()

        # Log results
        print(f"Stage2 Epoch {epoch + 1}/{DINOV2_TRAIN_CONFIG['stage2_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f}, "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}")

        # Save checkpoint
        is_best = val_mae < stage2_best_mae
        if is_best:
            stage2_best_mae = val_mae

        save_checkpoint(
            model, optimizer2, scheduler2, epoch,
            {'best_val_mae': stage2_best_mae}, DINOV2_TRAIN_CONFIG,
            checkpoint_dir, 'stage2', is_best
        )

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        # Early stopping
        if len(history['val_mae']) >= DINOV2_TRAIN_CONFIG['early_stopping_patience'] + DINOV2_TRAIN_CONFIG['stage1_epochs']:
            recent_maes = history['val_mae'][-DINOV2_TRAIN_CONFIG['early_stopping_patience']:]
            if all(recent_maes[i] >= recent_maes[i+1] for i in range(len(recent_maes)-1)):
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Final visualization
    visualize_training_progress(
        history,
        save_path=os.path.join(experiment_dir, 'training_progress.png')
    )

    print(f"\nTraining completed!")
    print(f"Stage 1 best MAE: {best_val_mae:.2f}")
    print(f"Stage 2 best MAE: {stage2_best_mae:.2f}")
    print(f"Final improvement: {best_val_mae - stage2_best_mae:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DINOv2 TransCrowd model')
    parser.add_argument('--name', type=str, default=None, help='experiment name')
    args = parser.parse_args()

    train_dinov2(args)