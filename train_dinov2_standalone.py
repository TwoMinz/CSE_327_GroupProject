"""
독립적인 DINOv2 훈련 스크립트 - 모든 필요한 코드 포함
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import scipy.io as sio
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime


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

# DINOv2 최적화 설정
DINOV2_MODEL_CONFIG = {
    'backbone': 'dinov2',
    'img_size': 392,
    'dinov2_size': 'base',
    'pretrained': True,
    'freeze_backbone': True,
    'dropout_rate': 0.1,
}

DINOV2_TRAIN_CONFIG = {
    'part': 'A',
    'batch_size': 16,
    'stage1_epochs': 30,
    'stage2_epochs': 120,
    'stage1_learning_rate': 1e-3,
    'stage1_weight_decay': 1e-4,
    'stage2_learning_rate': 1e-5,
    'stage2_weight_decay': 1e-4,
    'lr_scheduler': 'cosine',
    'warmup_epochs': 5,
    'early_stopping_patience': 20,
    'grad_clip_norm': 0.5,
    'seed': 42,
    'num_workers': 4,
    'pin_memory': True,
    'log_freq': 10,
}


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


# Dataset class
class ShanghaiTechDataset(Dataset):
    """Dataset class for ShanghaiTech crowd counting dataset."""

    def __init__(self, part='A', split='train', transform=None):
        self.part = part
        self.split = split
        self.transform = transform

        if part == 'A':
            self.root_dir = PART_A_DIR
        else:
            raise ValueError("Only part A is configured in this script")

        # Find image directory
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
                print(f"Found images directory: {self.img_dir}")
                break

        if self.img_dir is None:
            print(f"WARNING: Could not find images directory for part {part}, split {split}!")
            self.img_dir = possible_img_dirs[0]

        # Find ground truth directory
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
                print(f"Found ground truth directory: {self.gt_dir}")
                break

        if self.gt_dir is None:
            print(f"WARNING: Could not find ground truth directory!")
            self.gt_dir = possible_gt_dirs[0]

        # Get image files
        img_extensions = ['*.jpg', '*.jpeg', '*.png']
        self.img_paths = []
        for ext in img_extensions:
            self.img_paths.extend(glob.glob(os.path.join(self.img_dir, ext)))

        if not self.img_paths:
            print(f"WARNING: No images found!")
        else:
            print(f"Found {len(self.img_paths)} images for part {part}, split {split}")

        self.img_paths.sort()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        # Get ground truth
        img_name = os.path.basename(img_path)
        img_id = img_name.split('.')[0]

        gt_patterns = [
            f'GT_{img_id}.mat',
            f'{img_id}.mat',
            f'GT_{img_id.lower()}.mat',
            f'GT_{img_id.upper()}.mat'
        ]

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
            except Exception as e:
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


# DINOv2 Model
class DINOv2Backbone(nn.Module):
    """DINOv2 backbone for feature extraction."""

    def __init__(self, model_size='base', pretrained=True, img_size=392, freeze_backbone=False):
        super(DINOv2Backbone, self).__init__()

        self.model_size = model_size
        self.img_size = img_size
        self.freeze_backbone = freeze_backbone

        self.size_configs = {
            'small': {'embed_dim': 384, 'model_name': 'dinov2_vits14'},
            'base': {'embed_dim': 768, 'model_name': 'dinov2_vitb14'},
            'large': {'embed_dim': 1024, 'model_name': 'dinov2_vitl14'},
            'giant': {'embed_dim': 1536, 'model_name': 'dinov2_vitg14'}
        }

        self.config = self.size_configs[model_size]
        self.embed_dim = self.config['embed_dim']

        if pretrained:
            try:
                self.backbone = torch.hub.load('facebookresearch/dinov2', self.config['model_name'])
                print(f"✓ Loaded pretrained DINOv2 {model_size} model")
            except Exception as e:
                print(f"Failed to load pretrained model: {e}")
                self.backbone = self._create_basic_transformer()
        else:
            self.backbone = self._create_basic_transformer()

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ Backbone weights frozen")

        self.num_features = self.embed_dim

    def _create_basic_transformer(self):
        """Create basic transformer as fallback"""
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
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
        else:
            if self.freeze_backbone:
                with torch.no_grad():
                    features = self.backbone(x)
            else:
                features = self.backbone(x)

        # Debug: print the type and keys of features
        # print(f"Debug - features type: {type(features)}")
        # if isinstance(features, dict):
        #     print(f"Debug - features keys: {list(features.keys())}")

        # Handle different output formats more robustly
        if isinstance(features, dict):
            # Try common keys for different model outputs
            if 'x' in features:
                features = features['x']
            elif 'last_hidden_state' in features:
                features = features['last_hidden_state']
            elif 'pooler_output' in features:
                features = features['pooler_output']
            elif 'hidden_states' in features:
                features = features['hidden_states']
            else:
                # If none of the expected keys exist, try to get the first available tensor
                print(f"Warning: Unexpected dict keys: {list(features.keys())}")
                for key, value in features.items():
                    if isinstance(value, torch.Tensor):
                        features = value
                        print(f"Using key '{key}' as features")
                        break
                else:
                    raise ValueError(f"No tensor found in features dict with keys: {list(features.keys())}")
        elif isinstance(features, tuple):
            features = features[0]
        elif isinstance(features, list):
            features = features[0]

        # Handle tensor dimensions
        if isinstance(features, torch.Tensor):
            if len(features.shape) == 3:  # (B, N, D) format
                features = features[:, 0, :]  # Take CLS token
            elif len(features.shape) == 4:  # (B, C, H, W) format
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
            elif len(features.shape) == 2:  # (B, D) format - already good
                pass
            else:
                raise ValueError(f"Unexpected tensor shape: {features.shape}")
        else:
            raise ValueError(f"Features is not a tensor: {type(features)}")

        return features


class DINOv2WithRegression(nn.Module):
    """DINOv2 backbone with regression head."""

    def __init__(self, model_size='base', pretrained=True, img_size=392,
                 dropout_rate=0.1, freeze_backbone=False):
        super(DINOv2WithRegression, self).__init__()

        self.backbone = DINOv2Backbone(
            model_size=model_size,
            pretrained=pretrained,
            img_size=img_size,
            freeze_backbone=freeze_backbone
        )

        self.regression_head = nn.Sequential(
            nn.LayerNorm(self.backbone.num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

        self._init_regression_head()

    def _init_regression_head(self):
        """Initialize regression head weights."""
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
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
    """TransCrowd model with DINOv2 backbone."""

    def __init__(self, config):
        super(TransCrowd, self).__init__()
        self.config = config

        self.model = DINOv2WithRegression(
            model_size=config.get('dinov2_size', 'base'),
            pretrained=config.get('pretrained', True),
            img_size=config.get('img_size', 392),
            dropout_rate=config.get('dropout_rate', 0.1),
            freeze_backbone=config.get('freeze_backbone', False)
        )

    def forward(self, x):
        return self.model(x)


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


def get_dinov2_transforms(img_size=392):
    """DINOv2 optimized transforms"""
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
    """Get dataloaders"""
    train_transform, test_transform = get_dinov2_transforms(config['img_size'])

    train_dataset = ShanghaiTechDataset(part=config['part'], split='train', transform=train_transform)
    test_dataset = ShanghaiTechDataset(part=config['part'], split='test', transform=test_transform)

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
    """Get optimizer and scheduler for each stage"""
    if stage == 'stage1':
        lr = config['stage1_learning_rate']
        weight_decay = config['stage1_weight_decay']

        # Freeze backbone
        for param in model.model.backbone.parameters():
            param.requires_grad = False
        trainable_params = model.model.regression_head.parameters()

    else:  # stage2
        lr = config['stage2_learning_rate']
        weight_decay = config['stage2_weight_decay']

        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True

        # Different learning rates
        trainable_params = [
            {'params': model.model.backbone.parameters(), 'lr': lr * 0.1},
            {'params': model.model.regression_head.parameters(), 'lr': lr}
        ]

    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, weight_decay=weight_decay,
        betas=(0.9, 0.999), eps=1e-8
    )

    if config['lr_scheduler'] == 'cosine':
        total_epochs = config['stage1_epochs'] if stage == 'stage1' else config['stage2_epochs']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - config['warmup_epochs'], eta_min=lr * 0.01
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


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train DINOv2 TransCrowd model')
    parser.add_argument('--name', type=str, default=None, help='experiment name')
    parser.add_argument('--size', type=str, default='base', choices=['small', 'base', 'large'], help='DINOv2 model size')
    args = parser.parse_args()

    # Update config with model size
    DINOV2_MODEL_CONFIG['dinov2_size'] = args.size

    # Set seed
    set_seed(DINOV2_TRAIN_CONFIG['seed'])

    # Setup directories
    experiment_name = f"dinov2_{args.size}_optimized" if args.name is None else args.name
    experiment_dir = os.path.join(LOG_DIR, experiment_name)
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, experiment_name)

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Experiment: {experiment_name}")
    print(f"DINOv2 size: {args.size}")

    # Get dataloaders
    train_loader, val_loader = get_dataloaders({**DINOV2_TRAIN_CONFIG, **DINOV2_MODEL_CONFIG})

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

    # Stage 1
    optimizer1, scheduler1 = get_optimizer_and_scheduler(model, DINOV2_TRAIN_CONFIG, 'stage1')
    print_model_info(model)

    for epoch in range(DINOV2_TRAIN_CONFIG['stage1_epochs']):
        train_loss, train_mae = train_one_epoch(
            model, train_loader, criterion, optimizer1, DEVICE, epoch, DINOV2_TRAIN_CONFIG, 'Stage1'
        )

        val_loss, val_mae = validate(model, val_loader, criterion, DEVICE)

        if scheduler1:
            scheduler1.step()

        print(f"Stage1 Epoch {epoch + 1}/{DINOV2_TRAIN_CONFIG['stage1_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f}, "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}")

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae

        save_checkpoint(model, optimizer1, scheduler1, epoch,
                       {'best_val_mae': best_val_mae}, DINOV2_TRAIN_CONFIG,
                       checkpoint_dir, 'stage1', is_best)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

    print(f"\nStage 1 completed. Best validation MAE: {best_val_mae:.2f}")

    print("\n" + "="*80)
    print("STAGE 2: Fine-tuning Full Model")
    print("="*80)

    # Stage 2
    optimizer2, scheduler2 = get_optimizer_and_scheduler(model, DINOV2_TRAIN_CONFIG, 'stage2')
    print_model_info(model)

    stage2_best_mae = best_val_mae

    for epoch in range(DINOV2_TRAIN_CONFIG['stage2_epochs']):
        train_loss, train_mae = train_one_epoch(
            model, train_loader, criterion, optimizer2, DEVICE, epoch, DINOV2_TRAIN_CONFIG, 'Stage2'
        )

        val_loss, val_mae = validate(model, val_loader, criterion, DEVICE)

        if scheduler2:
            scheduler2.step()

        print(f"Stage2 Epoch {epoch + 1}/{DINOV2_TRAIN_CONFIG['stage2_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f}, "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}")

        is_best = val_mae < stage2_best_mae
        if is_best:
            stage2_best_mae = val_mae

        save_checkpoint(model, optimizer2, scheduler2, epoch,
                       {'best_val_mae': stage2_best_mae}, DINOV2_TRAIN_CONFIG,
                       checkpoint_dir, 'stage2', is_best)

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

    print(f"\nTraining completed!")
    print(f"Stage 1 best MAE: {best_val_mae:.2f}")
    print(f"Stage 2 best MAE: {stage2_best_mae:.2f}")
    print(f"Final improvement: {best_val_mae - stage2_best_mae:.2f}")


if __name__ == '__main__':
    main()