"""
Cross-dataset evaluation script for TransCrowd models with different vision transformer backbones.
Tests multiple trained models on various crowd counting datasets.
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import scipy.io as sio
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'outputs', 'checkpoints')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'cross_dataset_evaluation')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset configurations
DATASET_CONFIGS = {
    'ShanghaiTech_A': {
        'path': os.path.join(DATA_DIR, 'ShanghaiTech', 'part_A'),
        'test_split': 'test',
        'image_dirs': ['test_data/images', 'test/images', 'images/test', 'test', 'images'],
        'gt_dirs': ['test_data/ground-truth', 'test_data/ground_truth', 'test_data/groundtruth',
                   'test/ground-truth', 'ground-truth', 'ground_truth', 'groundtruth'],
        'gt_key_patterns': ['image_info', 'annPoints', 'points'],
        'description': 'ShanghaiTech Part A - Dense crowd scenes'
    },
    'ShanghaiTech_B': {
        'path': os.path.join(DATA_DIR, 'ShanghaiTech', 'part_B'),
        'test_split': 'test',
        'image_dirs': ['test_data/images', 'test/images', 'images/test', 'test', 'images'],
        'gt_dirs': ['test_data/ground-truth', 'test_data/ground_truth', 'test_data/groundtruth',
                   'test/ground-truth', 'ground-truth', 'ground_truth', 'groundtruth'],
        'gt_key_patterns': ['image_info', 'annPoints', 'points'],
        'description': 'ShanghaiTech Part B - Sparse crowd scenes'
    },
    'UCF_QNRF': {
        'path': os.path.join(DATA_DIR, 'UCF-QNRF_ECCV18'),
        'test_split': 'Test',
        'image_dirs': ['Test', 'test', 'Test_Images', 'test_images'],
        'gt_dirs': ['Test_Images', 'Test', 'test', 'ground_truth', 'groundtruth'],
        'gt_key_patterns': ['annPoints', 'points', 'image_info'],
        'description': 'UCF-QNRF - High resolution diverse scenes'
    },
    'JHU_CROWD': {
        'path': os.path.join(DATA_DIR, 'jhu_crowd_v2.0'),
        'test_split': 'test',
        'image_dirs': ['test/images', 'test', 'images/test', 'images'],
        'gt_dirs': ['test/gt', 'test/ground_truth', 'test/groundtruth', 'gt', 'ground_truth'],
        'gt_key_patterns': ['points', 'annPoints', 'image_info'],
        'description': 'JHU-CROWD++ - Weather and lighting variations'
    },
    'NWPU_CROWD': {
        'path': os.path.join(DATA_DIR, 'NWPU-Crowd'),
        'test_split': 'test',
        'image_dirs': ['test_data/images', 'test/images', 'images/test', 'test', 'images'],
        'gt_dirs': ['test_data/ground_truth', 'test/ground_truth', 'test/gt', 'ground_truth', 'gt'],
        'gt_key_patterns': ['points', 'annPoints', 'image_info'],
        'description': 'NWPU-Crowd - Large scale crowd scenes'
    }
}

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_available_datasets():
    """Find which datasets are available in the data directory."""
    available = []
    for dataset_name, config in DATASET_CONFIGS.items():
        if os.path.exists(config['path']):
            available.append(dataset_name)
            print(f"✓ Found dataset: {dataset_name} at {config['path']}")
        else:
            print(f"✗ Dataset not found: {dataset_name} at {config['path']}")
    return available

def find_available_models():
    """Find all available trained models in checkpoint directory."""
    available_models = []

    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
        return available_models

    for model_dir in os.listdir(CHECKPOINT_DIR):
        model_path = os.path.join(CHECKPOINT_DIR, model_dir)
        if os.path.isdir(model_path):
            # Look for checkpoint files
            checkpoint_files = []
            for checkpoint_name in ['best_stage2.pth', 'latest_stage2.pth', 'best_stage1.pth', 'latest_stage1.pth']:
                checkpoint_path = os.path.join(model_path, checkpoint_name)
                if os.path.exists(checkpoint_path):
                    checkpoint_files.append(checkpoint_name)

            if checkpoint_files:
                available_models.append({
                    'name': model_dir,
                    'path': model_path,
                    'checkpoints': checkpoint_files
                })
                print(f"✓ Found model: {model_dir} with checkpoints: {checkpoint_files}")

    return available_models

class UniversalCrowdDataset(Dataset):
    """Universal dataset class that can handle different crowd counting datasets."""

    def __init__(self, dataset_name, transform=None, max_samples=None):
        self.dataset_name = dataset_name
        self.transform = transform
        self.config = DATASET_CONFIGS[dataset_name]

        print(f"Loading dataset: {dataset_name}")

        # Find image directory
        self.img_dir = self._find_directory(self.config['image_dirs'])
        if not self.img_dir:
            raise ValueError(f"Could not find image directory for {dataset_name}")

        # Find ground truth directory
        self.gt_dir = self._find_directory(self.config['gt_dirs'])
        if not self.gt_dir:
            print(f"Warning: Could not find ground truth directory for {dataset_name}")
            self.gt_dir = None

        # Get image files
        self.img_paths = self._get_image_files()

        if max_samples and max_samples < len(self.img_paths):
            self.img_paths = self.img_paths[:max_samples]
            print(f"Limited to {max_samples} samples for faster evaluation")

        print(f"Found {len(self.img_paths)} images for {dataset_name}")

    def _find_directory(self, possible_dirs):
        """Find the first existing directory from a list of possibilities."""
        base_path = self.config['path']
        for dir_name in possible_dirs:
            full_path = os.path.join(base_path, dir_name)
            if os.path.exists(full_path):
                return full_path
        return None

    def _get_image_files(self):
        """Get all image files from the image directory."""
        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        img_paths = []

        for ext in img_extensions:
            img_paths.extend(glob.glob(os.path.join(self.img_dir, ext)))
            img_paths.extend(glob.glob(os.path.join(self.img_dir, ext.upper())))

        return sorted(img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a dummy image
            img = Image.new('RGB', (224, 224), color='black')

        # Get ground truth count
        count = self._get_ground_truth_count(img_path)

        # Apply transform
        if self.transform:
            try:
                img = self.transform(img)
            except Exception as e:
                print(f"Error applying transform to {img_path}: {e}")
                # Create dummy tensor
                img = torch.zeros(3, 224, 224)

        return {
            'image': img,
            'count': torch.tensor(count, dtype=torch.float),
            'path': img_path,
            'img_name': os.path.basename(img_path)
        }

    def _get_ground_truth_count(self, img_path):
        """Extract ground truth count for an image."""
        if not self.gt_dir:
            return 0.0

        img_name = os.path.basename(img_path)
        img_id = os.path.splitext(img_name)[0]

        # Try different ground truth file naming patterns
        gt_patterns = [
            f'GT_{img_id}.mat',
            f'{img_id}.mat',
            f'{img_id}_GT.mat',
            f'GT_{img_id.lower()}.mat',
            f'GT_{img_id.upper()}.mat',
            f'{img_id.lower()}.mat',
            f'{img_id.upper()}.mat'
        ]

        for pattern in gt_patterns:
            gt_path = os.path.join(self.gt_dir, pattern)
            if os.path.exists(gt_path):
                try:
                    return self._extract_count_from_mat(gt_path)
                except Exception as e:
                    print(f"Error reading ground truth {gt_path}: {e}")
                    continue

        print(f"Warning: No ground truth found for {img_name}")
        return 0.0

    def _extract_count_from_mat(self, gt_path):
        """Extract count from a .mat ground truth file."""
        try:
            gt_data = sio.loadmat(gt_path)

            # Try different key patterns
            for key_pattern in self.config['gt_key_patterns']:
                if key_pattern == 'image_info':
                    try:
                        points = gt_data['image_info'][0, 0]['location'][0, 0]
                        return float(points.shape[0])
                    except (KeyError, IndexError):
                        continue
                elif key_pattern in gt_data:
                    points = gt_data[key_pattern]
                    if isinstance(points, np.ndarray):
                        return float(points.shape[0])

            # If no standard key found, look for any array that could be points
            for key, value in gt_data.items():
                if isinstance(value, np.ndarray) and len(value.shape) == 2 and value.shape[1] >= 2:
                    return float(value.shape[0])

            return 0.0

        except Exception as e:
            print(f"Error extracting count from {gt_path}: {e}")
            return 0.0

def get_test_transforms(img_size=392):
    """Get test transforms."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def custom_collate_fn(batch):
    """Custom collate function."""
    images = [item['image'] for item in batch]
    counts = [item['count'] for item in batch]
    paths = [item['path'] for item in batch]
    names = [item['img_name'] for item in batch]

    return {
        'image': torch.stack(images, 0),
        'count': torch.stack(counts, 0),
        'path': paths,
        'img_name': names
    }

# Model classes (same as before but with better error handling)
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

        self.config = self.size_configs.get(model_size, self.size_configs['base'])
        self.embed_dim = self.config['embed_dim']
        self.use_fallback = False

        if pretrained:
            try:
                self.backbone = torch.hub.load('facebookresearch/dinov2', self.config['model_name'])
            except Exception as e:
                print(f"Using fallback model: {e}")
                self.backbone = self._create_basic_transformer()
                self.use_fallback = True
        else:
            self.backbone = self._create_basic_transformer()
            self.use_fallback = True

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
                # Handle different DINOv2 output formats
                if 'x_norm_clstoken' in features:
                    features = features['x_norm_clstoken']
                elif 'x_norm_patchtokens' in features:
                    patch_tokens = features['x_norm_patchtokens']
                    features = torch.mean(patch_tokens, dim=1)
                else:
                    for key, value in features.items():
                        if isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                            features = value
                            break

                if isinstance(features, torch.Tensor) and len(features.shape) == 3:
                    features = features[:, 0, :]

            return features
        except Exception:
            self.use_fallback = True
            self.backbone = self._create_basic_transformer().to(x.device)
            return self.backbone(x)

class DINOv2WithRegression(nn.Module):
    def __init__(self, model_size='base', pretrained=True, img_size=392, dropout_rate=0.1, freeze_backbone=False, enhanced_head=False):
        super(DINOv2WithRegression, self).__init__()

        self.backbone = DINOv2Backbone(model_size=model_size, pretrained=pretrained,
                                       img_size=img_size, freeze_backbone=freeze_backbone)

        # Create regression head based on enhanced_head flag
        if enhanced_head:
            # Enhanced regression head with additional layer (4-layer version)
            self.regression_head = nn.Sequential(
                nn.LayerNorm(self.backbone.num_features),
                nn.Dropout(dropout_rate),
                nn.Linear(self.backbone.num_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 64),  # Additional layer
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(64, 1)     # Final layer
            )
        else:
            # Standard regression head (3-layer version)
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

    def forward(self, x):
        features = self.backbone(x)
        count = self.regression_head(features)
        return count.squeeze(-1)

class TransCrowd(nn.Module):
    def __init__(self, config):
        super(TransCrowd, self).__init__()
        self.config = config

        # Check if enhanced head is needed based on config
        enhanced_head = config.get('enhanced_head', False)

        self.model = DINOv2WithRegression(
            model_size=config.get('dinov2_size', 'base'),
            pretrained=config.get('pretrained', True),
            img_size=config.get('img_size', 392),
            dropout_rate=config.get('dropout_rate', 0.1),
            freeze_backbone=config.get('freeze_backbone', False),
            enhanced_head=enhanced_head
        )

    def forward(self, x):
        return self.model(x)

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint with automatic architecture detection."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint with fallbacks
    config = checkpoint.get('config', {
        'dinov2_size': 'base',
        'pretrained': True,
        'img_size': 392,
        'dropout_rate': 0.1,
        'freeze_backbone': False
    })

    # Auto-detect if enhanced head is needed by checking state_dict
    state_dict = checkpoint['model_state_dict']
    enhanced_head = False

    # Check for enhanced head signature (layer 11 exists)
    if any('regression_head.11.' in key for key in state_dict.keys()):
        enhanced_head = True
        config['enhanced_head'] = True
        print(f"  Detected enhanced regression head (4-layer)")
    else:
        print(f"  Detected standard regression head (3-layer)")

    # Create model with appropriate architecture
    model = TransCrowd(config)

    try:
        model.load_state_dict(state_dict)
        print(f"  ✓ State dict loaded successfully")
    except RuntimeError as e:
        # If loading fails, try to load with strict=False
        print(f"  ⚠️ Partial loading due to architecture mismatch: {e}")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"    Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"    Unexpected keys: {unexpected_keys}")

        # If too many keys are missing, this model might not be usable
        if len(missing_keys) > 5:
            raise RuntimeError(f"Too many missing keys ({len(missing_keys)}), model incompatible")

    model = model.to(device)
    model.eval()

    return model, config

def evaluate_model_on_dataset(model, dataset_name, config, batch_size=16, max_samples=None):
    """Evaluate a model on a specific dataset."""
    print(f"\nEvaluating on {dataset_name}...")

    # Create dataset
    transform = get_test_transforms(config.get('img_size', 392))

    try:
        dataset = UniversalCrowdDataset(dataset_name, transform=transform, max_samples=max_samples)
    except Exception as e:
        print(f"Error creating dataset {dataset_name}: {e}")
        return None

    if len(dataset) == 0:
        print(f"No images found in {dataset_name}")
        return None

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=custom_collate_fn, pin_memory=True
    )

    # Evaluate
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            try:
                images = batch['image'].to(DEVICE)
                targets = batch['count'].to(DEVICE)

                predictions = model(images)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue

    if not all_predictions:
        print(f"No successful predictions for {dataset_name}")
        return None

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate metrics
    mae = np.mean(np.abs(all_predictions - all_targets))
    mse = np.mean((all_predictions - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((all_predictions - all_targets) / (all_targets + 1e-4))) * 100
    correlation = np.corrcoef(all_predictions, all_targets)[0, 1] if len(all_predictions) > 1 else 0

    return {
        'dataset': dataset_name,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'correlation': correlation,
        'num_samples': len(all_predictions),
        'predictions': all_predictions,
        'targets': all_targets
    }

def create_comparison_visualizations(all_results, save_dir):
    """Create comprehensive comparison visualizations."""
    os.makedirs(save_dir, exist_ok=True)

    # Prepare data for visualization
    models = list(all_results.keys())
    datasets = list(all_results[models[0]].keys()) if models else []

    # 1. MAE Heatmap
    mae_data = []
    for model in models:
        mae_row = []
        for dataset in datasets:
            if dataset in all_results[model] and all_results[model][dataset] is not None:
                mae_row.append(all_results[model][dataset]['mae'])
            else:
                mae_row.append(np.nan)
        mae_data.append(mae_row)

    plt.figure(figsize=(12, 8))
    mae_df = pd.DataFrame(mae_data, index=models, columns=datasets)
    sns.heatmap(mae_df, annot=True, fmt='.2f', cmap='RdYlBu_r', center=50)
    plt.title('Mean Absolute Error (MAE) Comparison Across Models and Datasets')
    plt.ylabel('Models')
    plt.xlabel('Datasets')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mae_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Performance Summary Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = ['mae', 'mse', 'rmse', 'correlation']
    metric_names = ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error', 'Correlation']

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]

        metric_data = []
        model_names = []

        for model in models:
            for dataset in datasets:
                if dataset in all_results[model] and all_results[model][dataset] is not None:
                    metric_data.append(all_results[model][dataset][metric])
                    model_names.append(f"{model}\n({dataset})")

        if metric_data:
            bars = ax.bar(range(len(metric_data)), metric_data)
            ax.set_title(metric_name)
            ax.set_xticks(range(len(metric_data)))
            ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)

            # Color bars by model
            colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
            model_color_map = {model: colors[i] for i, model in enumerate(models)}

            for i, bar in enumerate(bars):
                model_name = model_names[i].split('\n')[0]
                bar.set_color(model_color_map[model_name])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Model ranking based on average performance
    model_avg_mae = {}
    for model in models:
        maes = []
        for dataset in datasets:
            if dataset in all_results[model] and all_results[model][dataset] is not None:
                maes.append(all_results[model][dataset]['mae'])
        model_avg_mae[model] = np.mean(maes) if maes else float('inf')

    # Sort models by average MAE
    sorted_models = sorted(model_avg_mae.items(), key=lambda x: x[1])

    plt.figure(figsize=(12, 8))
    models_sorted = [item[0] for item in sorted_models]
    avg_maes = [item[1] for item in sorted_models if item[1] != float('inf')]

    bars = plt.bar(range(len(avg_maes)), avg_maes)
    plt.title('Model Ranking by Average MAE Across All Datasets')
    plt.xlabel('Models (Best to Worst)')
    plt.ylabel('Average MAE')
    plt.xticks(range(len(avg_maes)), models_sorted[:len(avg_maes)], rotation=45, ha='right')

    # Color gradient
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(avg_maes)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Add value labels on bars
    for i, v in enumerate(avg_maes):
        plt.text(i, v + max(avg_maes) * 0.01, f'{v:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualizations saved to {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Cross-dataset evaluation of TransCrowd models')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Specific model names to evaluate (default: all available)')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to use (default: all available)')
    parser.add_argument('--checkpoint-priority', choices=['best_stage2', 'latest_stage2', 'best_stage1', 'latest_stage1'],
                       default='best_stage2', help='Which checkpoint to prioritize')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples per dataset (for quick testing)')
    parser.add_argument('--save-detailed-results', action='store_true',
                       help='Save detailed prediction results')

    args = parser.parse_args()

    print("🧪 Cross-Dataset Evaluation of TransCrowd Models")
    print("=" * 60)
    print(f"Using device: {DEVICE}")

    # Set seed for reproducibility
    set_seed(42)

    # Find available datasets and models
    available_datasets = find_available_datasets()
    available_models = find_available_models()

    if not available_datasets:
        print("❌ No datasets found! Please check your data directory structure.")
        return

    if not available_models:
        print("❌ No trained models found! Please check your checkpoint directory.")
        return

    # Filter datasets and models based on arguments
    datasets_to_test = args.datasets if args.datasets else available_datasets
    models_to_test = args.models if args.models else [model['name'] for model in available_models]

    print(f"\n📊 Will evaluate {len(models_to_test)} models on {len(datasets_to_test)} datasets")
    print(f"Models: {models_to_test}")
    print(f"Datasets: {datasets_to_test}")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_output_dir = os.path.join(OUTPUT_DIR, f"evaluation_{timestamp}")
    os.makedirs(eval_output_dir, exist_ok=True)

    # Main evaluation loop
    all_results = {}

    for model_info in available_models:
        model_name = model_info['name']

        if model_name not in models_to_test:
            continue

        print(f"\n🔧 Loading model: {model_name}")

        # Find the best available checkpoint
        checkpoint_path = None
        for checkpoint_name in [args.checkpoint_priority] + ['best_stage2.pth', 'latest_stage2.pth', 'best_stage1.pth', 'latest_stage1.pth']:
            potential_path = os.path.join(model_info['path'], checkpoint_name)
            if os.path.exists(potential_path):
                checkpoint_path = potential_path
                print(f"  Using checkpoint: {checkpoint_name}")
                break

        if not checkpoint_path:
            print(f"  ❌ No valid checkpoint found for {model_name}")
            continue

        # Load model
        try:
            model, config = load_model(checkpoint_path, DEVICE)
            print(f"  ✓ Model loaded successfully")
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
            continue

        # Evaluate on all datasets
        model_results = {}

        for dataset_name in datasets_to_test:
            if dataset_name not in available_datasets:
                print(f"  ⚠️ Dataset {dataset_name} not available, skipping...")
                continue

            try:
                result = evaluate_model_on_dataset(
                    model, dataset_name, config,
                    batch_size=args.batch_size,
                    max_samples=args.max_samples
                )

                if result is not None:
                    model_results[dataset_name] = result
                    print(f"  ✓ {dataset_name}: MAE={result['mae']:.2f}, Samples={result['num_samples']}")
                else:
                    print(f"  ❌ Failed to evaluate on {dataset_name}")

            except Exception as e:
                print(f"  ❌ Error evaluating {dataset_name}: {e}")
                continue

        all_results[model_name] = model_results

        # Clear GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Create summary results
    print(f"\n📋 Creating evaluation summary...")

    # Save detailed results
    summary_data = []
    for model_name, model_results in all_results.items():
        for dataset_name, result in model_results.items():
            summary_data.append({
                'Model': model_name,
                'Dataset': dataset_name,
                'Dataset_Description': DATASET_CONFIGS[dataset_name]['description'],
                'MAE': result['mae'],
                'MSE': result['mse'],
                'RMSE': result['rmse'],
                'MAPE': result['mape'],
                'Correlation': result['correlation'],
                'Num_Samples': result['num_samples']
            })

    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(eval_output_dir, 'evaluation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary saved to: {summary_path}")

    # Create detailed results if requested
    if args.save_detailed_results:
        detailed_dir = os.path.join(eval_output_dir, 'detailed_results')
        os.makedirs(detailed_dir, exist_ok=True)

        for model_name, model_results in all_results.items():
            for dataset_name, result in model_results.items():
                detailed_data = pd.DataFrame({
                    'prediction': result['predictions'],
                    'ground_truth': result['targets'],
                    'absolute_error': np.abs(result['predictions'] - result['targets']),
                    'relative_error': np.abs(result['predictions'] - result['targets']) / (result['targets'] + 1e-4)
                })

                detailed_path = os.path.join(detailed_dir, f'{model_name}_{dataset_name}_detailed.csv')
                detailed_data.to_csv(detailed_path, index=False)

        print(f"✓ Detailed results saved to: {detailed_dir}")

    # Create visualizations
    print(f"📊 Creating comparison visualizations...")
    viz_dir = os.path.join(eval_output_dir, 'visualizations')
    create_comparison_visualizations(all_results, viz_dir)

    # Print final summary
    print(f"\n🎉 Evaluation completed!")
    print(f"📁 Results saved to: {eval_output_dir}")
    print(f"\n📊 FINAL RESULTS SUMMARY:")
    print("=" * 80)

    # Find best model for each dataset
    for dataset_name in datasets_to_test:
        if dataset_name not in available_datasets:
            continue

        dataset_results = []
        for model_name, model_results in all_results.items():
            if dataset_name in model_results:
                dataset_results.append((model_name, model_results[dataset_name]['mae']))

        if dataset_results:
            dataset_results.sort(key=lambda x: x[1])
            best_model, best_mae = dataset_results[0]
            print(f"{dataset_name:20} | Best: {best_model:20} (MAE: {best_mae:.2f})")

    # Find overall best model (average MAE across all datasets)
    print(f"\n🏆 OVERALL RANKING (Average MAE):")
    print("-" * 50)

    model_avg_scores = {}
    for model_name, model_results in all_results.items():
        maes = [result['mae'] for result in model_results.values()]
        if maes:
            model_avg_scores[model_name] = np.mean(maes)

    if model_avg_scores:
        sorted_models = sorted(model_avg_scores.items(), key=lambda x: x[1])
        for rank, (model_name, avg_mae) in enumerate(sorted_models, 1):
            datasets_tested = len(all_results[model_name])
            print(f"{rank:2d}. {model_name:25} | Avg MAE: {avg_mae:6.2f} | Datasets: {datasets_tested}")

        # Declare the winner
        winner_model, winner_mae = sorted_models[0]
        print(f"\n🎯 BEST OVERALL MODEL: {winner_model}")
        print(f"   Average MAE: {winner_mae:.2f}")
        print(f"   Tested on {len(all_results[winner_model])} datasets")

    # Save final ranking
    ranking_data = []
    for rank, (model_name, avg_mae) in enumerate(sorted_models, 1):
        ranking_data.append({
            'Rank': rank,
            'Model': model_name,
            'Average_MAE': avg_mae,
            'Datasets_Tested': len(all_results[model_name])
        })

    ranking_df = pd.DataFrame(ranking_data)
    ranking_path = os.path.join(eval_output_dir, 'model_ranking.csv')
    ranking_df.to_csv(ranking_path, index=False)

    # Create recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 30)

    if len(sorted_models) >= 3:
        print(f"✅ For best accuracy: Use '{sorted_models[0][0]}'")
        print(f"🔄 For balanced performance: Consider '{sorted_models[1][0]}'")
        print(f"⚡ For alternative option: Try '{sorted_models[2][0]}'")

    # Performance analysis
    if len(datasets_to_test) > 1:
        print(f"\n📈 CROSS-DATASET ANALYSIS:")
        print("-" * 30)

        for dataset_name in datasets_to_test:
            if dataset_name not in available_datasets:
                continue

            dataset_maes = []
            for model_name, model_results in all_results.items():
                if dataset_name in model_results:
                    dataset_maes.append(model_results[dataset_name]['mae'])

            if dataset_maes:
                avg_mae = np.mean(dataset_maes)
                std_mae = np.std(dataset_maes)
                difficulty = "Hard" if avg_mae > 100 else "Medium" if avg_mae > 50 else "Easy"
                print(f"{dataset_name:20} | Avg MAE: {avg_mae:6.2f} ± {std_mae:5.2f} | Difficulty: {difficulty}")

    print(f"\n✨ Evaluation completed successfully!")
    return eval_output_dir

if __name__ == '__main__':
    try:
        output_dir = main()
        print(f"\n📂 All results available at: {output_dir}")
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()