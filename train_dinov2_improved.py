"""
Overfitting을 해결하기 위한 개선된 DINOv2 훈련 스크립트
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import scipy.io as sio
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime

# Import experiment configurations
from improved_dinov2_configs import ALL_EXPERIMENTS, get_experiment_config, print_experiment_summary, \
    print_recommendations

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'ShanghaiTech')
PART_A_DIR = os.path.join(DATA_DIR, 'part_A')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mean_absolute_error(pred, target):
    """Calculate mean absolute error."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    return np.mean(np.abs(pred - target))


# Enhanced Dataset with strong augmentation
class ShanghaiTechDataset(torch.utils.data.Dataset):
    def __init__(self, part='A', split='train', transform=None, augmentation_level='normal'):
        self.part = part
        self.split = split
        self.transform = transform
        self.augmentation_level = augmentation_level

        if part == 'A':
            self.root_dir = PART_A_DIR
        else:
            raise ValueError("Only part A is configured")

        # Find directories (same as before)
        possible_img_dirs = [
            os.path.join(self.root_dir, f'{split}_data', 'images'),
            os.path.join(self.root_dir, f'{split}', 'images'),
            os.path.join(self.root_dir, 'images', f'{split}'),
            os.path.join(self.root_dir, f'{split}'),
            os.path.join(self.root_dir, 'images')
        ]

        self.img_dir = None
        for dir_path in possible_img_dirs:
            if os.path.exists(dir_path):
                self.img_dir = dir_path
                break

        possible_gt_dirs = [
            os.path.join(self.root_dir, f'{split}_data', 'ground-truth'),
            os.path.join(self.root_dir, f'{split}_data', 'ground_truth'),
            os.path.join(self.root_dir, f'{split}_data', 'groundtruth'),
            os.path.join(self.root_dir, f'{split}', 'ground-truth'),
            os.path.join(self.root_dir, 'ground-truth'),
            os.path.join(self.root_dir, 'ground_truth'),
            os.path.join(self.root_dir, 'groundtruth')
        ]

        self.gt_dir = None
        for dir_path in possible_gt_dirs:
            if os.path.exists(dir_path):
                self.gt_dir = dir_path
                break

        # Get image files
        img_extensions = ['*.jpg', '*.jpeg', '*.png']
        self.img_paths = []
        for ext in img_extensions:
            self.img_paths.extend(glob.glob(os.path.join(self.img_dir, ext)))
        self.img_paths.sort()

        print(f"Found {len(self.img_paths)} images for part {part}, split {split}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        # Get ground truth (same as before)
        img_name = os.path.basename(img_path)
        img_id = img_name.split('.')[0]

        gt_patterns = [f'GT_{img_id}.mat', f'{img_id}.mat', f'GT_{img_id.lower()}.mat', f'GT_{img_id.upper()}.mat']

        gt_path = None
        for pattern in gt_patterns:
            path = os.path.join(self.gt_dir, pattern)
            if os.path.exists(path):
                gt_path = path
                break

        if gt_path and os.path.exists(gt_path):
            try:
                gt_data = sio.loadmat(gt_path)
                points = None
                if 'image_info' in gt_data:
                    try:
                        points = gt_data['image_info'][0, 0]['location'][0, 0]
                    except (KeyError, IndexError):
                        pass
                elif 'annPoints' in gt_data:
                    points = gt_data['annPoints']
                elif 'points' in gt_data:
                    points = gt_data['points']

                if points is None:
                    points = np.zeros((0, 2))
                count = points.shape[0]
            except Exception:
                points = np.zeros((0, 2))
                count = 0
        else:
            points = np.zeros((0, 2))
            count = 0

        if self.transform:
            img = self.transform(img)

        return {
            'image': img,
            'count': torch.tensor(count, dtype=torch.float),
            'points': torch.tensor(points, dtype=torch.float) if points.size > 0 else torch.zeros((0, 2)),
            'path': img_path
        }


def custom_collate_fn(batch):
    """Custom collate function."""
    images = [item['image'] for item in batch]
    counts = [item['count'] for item in batch]
    paths = [item['path'] for item in batch]
    points = [item['points'] for item in batch]

    return {
        'image': torch.stack(images, 0),
        'count': torch.stack(counts, 0),
        'path': paths,
        'points': points
    }


def get_enhanced_transforms(img_size=392, augmentation_level='normal'):
    """Get enhanced transforms with different augmentation levels"""

    dinov2_mean = [0.485, 0.456, 0.406]
    dinov2_std = [0.229, 0.224, 0.225]

    if augmentation_level == 'strong':
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),  # 회전 추가
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 강화된 색상 변경
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # 약간의 이동
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),  # 원근 변환
            transforms.ToTensor(),
            transforms.Normalize(mean=dinov2_mean, std=dinov2_std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))  # Random erasing
        ])
    else:  # normal
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


# DINOv2 Model (same structure as before, but with improvements)
class DINOv2Backbone(nn.Module):
    def __init__(self, model_size='base', pretrained=True, img_size=392, freeze_backbone=False):
        super(DINOv2Backbone, self).__init__()

        self.model_size = model_size
        self.img_size = img_size
        self.freeze_backbone = freeze_backbone

        self.size_configs = {
            'small': {'embed_dim': 384, 'model_name': 'dinov2_vits14'},
            'base': {'embed_dim': 768, 'model_name': 'dinov2_vitb14'},
            'large': {'embed_dim': 1024, 'model_name': 'dinov2_vitl14'},
        }

        self.config = self.size_configs[model_size]
        self.embed_dim = self.config['embed_dim']
        self.use_fallback = False

        if pretrained:
            try:
                self.backbone = torch.hub.load('facebookresearch/dinov2', self.config['model_name'])
                print(f"✓ Loaded DINOv2 {model_size} model")
            except Exception as e:
                print(f"Using fallback model: {e}")
                self.backbone = self._create_basic_transformer()
                self.use_fallback = True
        else:
            self.backbone = self._create_basic_transformer()
            self.use_fallback = True

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ Backbone weights frozen")

        self.num_features = self.embed_dim

    def _create_basic_transformer(self):
        class BasicTransformer(nn.Module):
            def __init__(self, embed_dim, img_size):
                super().__init__()
                self.embed_dim = embed_dim
                patch_size = 14
                num_patches = (img_size // patch_size) ** 2

                self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
                self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
                self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
                self.norm = nn.LayerNorm(embed_dim)

                # Better initialization
                nn.init.trunc_normal_(self.pos_embed, std=0.02)
                nn.init.trunc_normal_(self.cls_token, std=0.02)

            def forward(self, x):
                B = x.shape[0]
                x = self.patch_embed(x)
                x = x.flatten(2).transpose(1, 2)
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x = x + self.pos_embed
                x = self.norm(x)
                return x[:, 0]

        return BasicTransformer(self.embed_dim, self.img_size)

    def forward(self, x):
        if self.use_fallback:
            return self.backbone(x)

        try:
            features = self.backbone(x)

            if isinstance(features, dict):
                if 'x_norm_clstoken' in features:
                    features = features['x_norm_clstoken']
                elif 'x_norm_patchtokens' in features:
                    patch_tokens = features['x_norm_patchtokens']
                    features = torch.mean(patch_tokens, dim=1)
                elif 'x_prenorm' in features:
                    prenorm_features = features['x_prenorm']
                    features = prenorm_features[:, 0, :] if len(prenorm_features.shape) == 3 else prenorm_features
                else:
                    for key, value in features.items():
                        if isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                            features = value
                            break

                if isinstance(features, torch.Tensor) and len(features.shape) == 3:
                    features = features[:, 0, :]

            return features
        except Exception as e:
            print(f"DINOv2 forward error: {e}, switching to fallback")
            self.use_fallback = True
            self.backbone = self._create_basic_transformer().to(x.device)
            return self.backbone(x)


class EnhancedDINOv2WithRegression(nn.Module):
    """Enhanced DINOv2 with better regularization"""

    def __init__(self, model_size='base', pretrained=True, img_size=392,
                 dropout_rate=0.1, freeze_backbone=False):
        super(EnhancedDINOv2WithRegression, self).__init__()

        self.backbone = DINOv2Backbone(model_size=model_size, pretrained=pretrained,
                                       img_size=img_size, freeze_backbone=freeze_backbone)

        # Enhanced regression head with better regularization
        self.regression_head = nn.Sequential(
            nn.LayerNorm(self.backbone.num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),  # 추가 레이어로 더 부드러운 학습
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),  # 마지막에는 약간 낮은 dropout
            nn.Linear(64, 1)
        )

        self._init_regression_head()

    def _init_regression_head(self):
        """Better initialization for regression head."""
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                # He initialization for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        features = self.backbone(x)
        count = self.regression_head(features)
        return count.squeeze(-1)


class TransCrowd(nn.Module):
    def __init__(self, config):
        super(TransCrowd, self).__init__()
        self.config = config

        self.model = EnhancedDINOv2WithRegression(
            model_size=config.get('dinov2_size', 'base'),
            pretrained=config.get('pretrained', True),
            img_size=config.get('img_size', 392),
            dropout_rate=config.get('dropout_rate', 0.1),
            freeze_backbone=config.get('freeze_backbone', False)
        )

    def forward(self, x):
        return self.model(x)


class LabelSmoothingLoss(nn.Module):
    """Label smoothing for count regression"""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        # Add small noise to targets for label smoothing effect
        if self.training and self.smoothing > 0:
            noise = torch.randn_like(target) * self.smoothing * target.std()
            target = target + noise

        return nn.functional.mse_loss(pred, target)


class ImprovedCountLoss(nn.Module):
    """Improved loss with multiple components"""

    def __init__(self, mse_weight=1.0, mae_weight=0.1, relative_weight=0.1,
                 label_smoothing=0.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.relative_weight = relative_weight
        self.label_smoothing = label_smoothing

        if label_smoothing > 0:
            self.mse_loss = LabelSmoothingLoss(label_smoothing)
        else:
            self.mse_loss = nn.MSELoss()

    def forward(self, pred_count, gt_count):
        # MSE Loss (with optional label smoothing)
        mse_loss = self.mse_loss(pred_count, gt_count)

        # MAE Loss
        mae_loss = torch.mean(torch.abs(pred_count - gt_count))

        # Relative Error
        relative_error = torch.mean(torch.abs(pred_count - gt_count) / (gt_count + 1.0))

        # Combined loss
        total_loss = (self.mse_weight * mse_loss +
                      self.mae_weight * mae_loss +
                      self.relative_weight * relative_error)

        return total_loss


def get_dataloaders(config):
    """Get enhanced dataloaders"""
    augmentation_level = config.get('data_augmentation', 'normal')
    train_transform, test_transform = get_enhanced_transforms(
        config['img_size'], augmentation_level
    )

    train_dataset = ShanghaiTechDataset(
        part=config['part'], split='train', transform=train_transform,
        augmentation_level=augmentation_level
    )
    test_dataset = ShanghaiTechDataset(
        part=config['part'], split='test', transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=config['pin_memory'],
        drop_last=True, collate_fn=custom_collate_fn
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=config['pin_memory'],
        collate_fn=custom_collate_fn
    )

    return train_loader, test_loader


def get_optimizer_and_scheduler(model, config, stage='stage1'):
    """Enhanced optimizer with better settings"""
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

        # Check if we should keep backbone frozen in stage 2
        if config.get('freeze_backbone_stage2', False):
            # Keep backbone frozen
            if hasattr(model, 'model') and hasattr(model.model, 'backbone'):
                for param in model.model.backbone.parameters():
                    param.requires_grad = False
                trainable_params = model.model.regression_head.parameters()
            else:
                trainable_params = [p for p in model.parameters() if p.requires_grad]
        else:
            # Unfreeze all parameters with different learning rates
            for param in model.parameters():
                param.requires_grad = True

            if hasattr(model, 'model'):
                trainable_params = [
                    {'params': model.model.backbone.parameters(), 'lr': lr * 0.1},
                    {'params': model.model.regression_head.parameters(), 'lr': lr}
                ]
            else:
                trainable_params = model.parameters()

    # Use AdamW with better settings
    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, weight_decay=weight_decay,
        betas=(0.9, 0.999), eps=1e-8
    )

    # Enhanced scheduler
    if config['lr_scheduler'] == 'cosine':
        total_epochs = config['stage1_epochs'] if stage == 'stage1' else config['stage2_epochs']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - config['warmup_epochs'], eta_min=lr * 0.01
        )
    else:
        scheduler = None

    return optimizer, scheduler


def mixup_data(x, y, alpha=1.0):
    """Mixup augmentation for training data"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config, stage):
    """Enhanced training loop with mixup"""
    model.train()
    total_loss = 0
    total_mae = 0

    use_mixup = config.get('mixup_alpha', 0) > 0 and stage == 'stage2'

    pbar = tqdm(train_loader, desc=f'{stage} Epoch {epoch + 1}')
    for i, batch in enumerate(pbar):
        images = batch['image'].to(device)
        counts = batch['count'].to(device)

        # Apply mixup if enabled
        if use_mixup:
            images, counts = mixup_data(images, counts, config['mixup_alpha'])

        predictions = model(images)
        loss = criterion(predictions, counts)

        optimizer.zero_grad()
        loss.backward()

        if config['grad_clip_norm'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])

        optimizer.step()

        total_loss += loss.item()
        mae = mean_absolute_error(predictions.detach(), counts.detach())
        total_mae += mae

        if i % config['log_freq'] == 0:
            pbar.set_postfix({'loss': f'{loss.item():.2f}', 'mae': f'{mae:.2f}'})

    return total_loss / len(train_loader), total_mae / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validation loop"""
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

    return total_loss / len(val_loader), total_mae / len(val_loader)


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


def enhanced_early_stopping(val_maes, patience, min_delta=0.5):
    """Enhanced early stopping with minimum improvement threshold"""
    if len(val_maes) < patience:
        return False

    # Check if there's been improvement in the last 'patience' epochs
    recent_maes = val_maes[-patience:]
    best_recent = min(recent_maes)
    current_mae = val_maes[-1]

    # Stop if no improvement of at least min_delta in recent epochs
    if current_mae > best_recent + min_delta:
        return True

    return False


def main():
    """Enhanced main training function"""
    parser = argparse.ArgumentParser(description='Train improved DINOv2 model')
    parser.add_argument('--experiment', type=int, required=True, choices=range(1, 7),
                        help='Experiment number (1-6)')
    parser.add_argument('--custom-name', type=str, default=None, help='Custom experiment name')

    args = parser.parse_args()

    # Get experiment configuration
    exp_config = get_experiment_config(args.experiment)
    model_config = exp_config['model_config']
    train_config = exp_config['train_config']

    # Set seed
    set_seed(train_config.get('seed', 42))

    # Setup directories
    experiment_name = args.custom_name if args.custom_name else exp_config['name']
    experiment_dir = os.path.join(LOG_DIR, experiment_name)
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, experiment_name)

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"🧪 실험 {args.experiment}: {exp_config['name']}")
    print(f"📝 설명: {exp_config['description']}")
    print(f"🔧 Using device: {DEVICE}")
    print(f"📂 실험 디렉토리: {experiment_dir}")

    # Get dataloaders
    combined_config = {**model_config, **train_config}
    train_loader, val_loader = get_dataloaders(combined_config)

    # Create model
    model = TransCrowd(model_config).to(DEVICE)
    print_model_info(model)

    # Enhanced loss function
    criterion = ImprovedCountLoss(
        mse_weight=1.0,
        mae_weight=0.1,
        relative_weight=0.1,
        label_smoothing=train_config.get('label_smoothing', 0.0)
    )

    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    best_val_mae = float('inf')

    print("\n" + "=" * 80)
    print("STAGE 1: Training Regression Head")
    print("=" * 80)

    # Stage 1
    optimizer1, scheduler1 = get_optimizer_and_scheduler(model, train_config, 'stage1')
    print_model_info(model)

    for epoch in range(train_config['stage1_epochs']):
        train_loss, train_mae = train_one_epoch(
            model, train_loader, criterion, optimizer1, DEVICE, epoch, train_config, 'Stage1'
        )

        val_loss, val_mae = validate(model, val_loader, criterion, DEVICE)

        if scheduler1:
            scheduler1.step()

        print(f"Stage1 Epoch {epoch + 1}/{train_config['stage1_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f}, "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}")

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae

        save_checkpoint(model, optimizer1, scheduler1, epoch,
                        {'best_val_mae': best_val_mae}, combined_config,
                        checkpoint_dir, 'stage1', is_best)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

    print(f"\nStage 1 completed. Best validation MAE: {best_val_mae:.2f}")

    print("\n" + "=" * 80)
    print("STAGE 2: Fine-tuning")
    print("=" * 80)

    # Stage 2
    optimizer2, scheduler2 = get_optimizer_and_scheduler(model, train_config, 'stage2')
    print_model_info(model)

    stage2_best_mae = best_val_mae
    stage2_val_maes = []

    for epoch in range(train_config['stage2_epochs']):
        train_loss, train_mae = train_one_epoch(
            model, train_loader, criterion, optimizer2, DEVICE, epoch, train_config, 'Stage2'
        )

        val_loss, val_mae = validate(model, val_loader, criterion, DEVICE)
        stage2_val_maes.append(val_mae)

        if scheduler2:
            scheduler2.step()

        print(f"Stage2 Epoch {epoch + 1}/{train_config['stage2_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f}, "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}")

        is_best = val_mae < stage2_best_mae
        if is_best:
            stage2_best_mae = val_mae

        save_checkpoint(model, optimizer2, scheduler2, epoch,
                        {'best_val_mae': stage2_best_mae}, combined_config,
                        checkpoint_dir, 'stage2', is_best)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        # Enhanced early stopping
        if enhanced_early_stopping(stage2_val_maes, train_config['early_stopping_patience']):
            print(f"Enhanced early stopping at epoch {epoch + 1}")
            break

    # Final results
    train_val_gap = history['train_mae'][-1] - history['val_mae'][-1]

    print(f"\n🎉 훈련 완료!")
    print(f"Stage 1 best MAE: {best_val_mae:.2f}")
    print(f"Stage 2 best MAE: {stage2_best_mae:.2f}")
    print(f"Final improvement: {best_val_mae - stage2_best_mae:.2f}")
    print(f"Final Train-Val gap: {train_val_gap:.2f}")

    if abs(train_val_gap) < 20:
        print("✅ Overfitting 해결 성공!")
    else:
        print("⚠️ 여전히 overfitting이 있습니다.")

    # Save training history
    history_path = os.path.join(experiment_dir, 'training_history.txt')
    with open(history_path, 'w') as f:
        f.write(f"Experiment: {exp_config['name']}\n")
        f.write(f"Description: {exp_config['description']}\n")
        f.write(f"Stage 1 best MAE: {best_val_mae:.2f}\n")
        f.write(f"Stage 2 best MAE: {stage2_best_mae:.2f}\n")
        f.write(f"Final improvement: {best_val_mae - stage2_best_mae:.2f}\n")
        f.write(f"Final Train-Val gap: {train_val_gap:.2f}\n")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print_experiment_summary()
        print_recommendations()
    else:
        main()