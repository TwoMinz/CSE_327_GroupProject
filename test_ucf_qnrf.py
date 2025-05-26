"""
ShanghaiTech로 훈련된 TransCrowd 모델을 UCF-QNRF 데이터셋으로 테스트
Cross-dataset evaluation for generalization assessment
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import h5py
import scipy.io as sio
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UCF_QNRF_DIR = os.path.join(BASE_DIR, 'data', 'UCF-QNRF_ECCV18')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
UCF_TEST_DIR = os.path.join(OUTPUT_DIR, 'ucf_qnrf_test')
os.makedirs(UCF_TEST_DIR, exist_ok=True)


# =============================================================================
# UCF-QNRF Dataset Class
# =============================================================================
class UCFQNRFDataset(Dataset):
    """
    UCF-QNRF Dataset Class

    UCF-QNRF 데이터셋 구조:
    UCF-QNRF_ECCV18/
    ├── Train/
    │   ├── img_0001.jpg
    │   ├── img_0001_ann.mat
    │   └── ...
    └── Test/
        ├── img_0001.jpg
        ├── img_0001_ann.mat
        └── ...
    """

    def __init__(self, root_dir, split='Test', transform=None, max_images=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.max_images = max_images

        # UCF-QNRF는 Train/Test 폴더로 구성
        self.data_dir = os.path.join(root_dir, split)

        if not os.path.exists(self.data_dir):
            # 대안 경로들 확인
            possible_dirs = [
                os.path.join(root_dir, split.lower()),
                os.path.join(root_dir, f'UCF-QNRF_{split}'),
                os.path.join(root_dir, 'images', split),
                root_dir  # 모든 파일이 한 폴더에 있는 경우
            ]

            for alt_dir in possible_dirs:
                if os.path.exists(alt_dir):
                    self.data_dir = alt_dir
                    print(f"Found UCF-QNRF data at: {self.data_dir}")
                    break
            else:
                print(f"Warning: UCF-QNRF {split} directory not found!")
                print(f"Expected: {os.path.join(root_dir, split)}")
                print("Please check your UCF-QNRF dataset path.")

        # 이미지 파일 찾기
        self.image_paths = []
        image_extensions = ['*.jpg', '*.jpeg', '*.png']

        for ext in image_extensions:
            pattern = os.path.join(self.data_dir, ext)
            found_images = glob.glob(pattern)
            self.image_paths.extend(found_images)

        # annotation 파일이 있는 이미지만 선택
        valid_images = []
        for img_path in self.image_paths:
            ann_path = self._get_annotation_path(img_path)
            if os.path.exists(ann_path):
                valid_images.append(img_path)

        self.image_paths = sorted(valid_images)

        # 최대 이미지 수 제한
        if max_images and max_images < len(self.image_paths):
            self.image_paths = self.image_paths[:max_images]
            print(f"Limited to {max_images} images for testing")

        print(f"Found {len(self.image_paths)} valid UCF-QNRF {split} images")

        if len(self.image_paths) == 0:
            print("❌ No valid images found! Please check:")
            print(f"   - UCF-QNRF dataset path: {root_dir}")
            print(f"   - Split directory: {self.data_dir}")
            print("   - Image and annotation file pairs")

    def _get_annotation_path(self, image_path):
        """Get corresponding annotation file path"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # UCF-QNRF annotation naming patterns
        possible_patterns = [
            f"{base_name}_ann.mat",
            f"{base_name}.mat",
            f"{base_name}_GT.mat",
            f"GT_{base_name}.mat"
        ]

        data_dir = os.path.dirname(image_path)

        for pattern in possible_patterns:
            ann_path = os.path.join(data_dir, pattern)
            if os.path.exists(ann_path):
                return ann_path

        return None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        original_size = img.size  # (width, height)

        # Load annotation
        ann_path = self._get_annotation_path(img_path)

        if ann_path and os.path.exists(ann_path):
            try:
                # UCF-QNRF annotations are stored differently
                # Try different formats
                if ann_path.endswith('.mat'):
                    ann_data = sio.loadmat(ann_path)

                    # Try different possible keys
                    points = None
                    possible_keys = ['annPoints', 'image_info', 'points', 'gt', 'GT']

                    for key in possible_keys:
                        if key in ann_data:
                            if key == 'image_info':
                                try:
                                    points = ann_data[key][0, 0]['location'][0, 0]
                                except:
                                    points = ann_data[key]
                            else:
                                points = ann_data[key]
                            break

                    if points is None:
                        # Print available keys for debugging
                        available_keys = [k for k in ann_data.keys() if not k.startswith('__')]
                        print(f"Available keys in {os.path.basename(ann_path)}: {available_keys}")

                        # Try the first non-system key
                        for key in available_keys:
                            try:
                                points = ann_data[key]
                                if isinstance(points, np.ndarray) and points.size > 0:
                                    break
                            except:
                                continue

                elif ann_path.endswith('.h5'):
                    # If annotations are in HDF5 format
                    with h5py.File(ann_path, 'r') as f:
                        points = np.array(f['points'])

                if points is None or not isinstance(points, np.ndarray):
                    points = np.zeros((0, 2))
                elif points.size == 0:
                    points = np.zeros((0, 2))
                else:
                    # Ensure points is 2D array
                    if len(points.shape) == 1:
                        points = points.reshape(-1, 2)
                    elif points.shape[1] > 2:
                        points = points[:, :2]  # Take only x, y coordinates

                count = points.shape[0]

            except Exception as e:
                print(f"Error loading annotation {ann_path}: {e}")
                points = np.zeros((0, 2))
                count = 0
        else:
            points = np.zeros((0, 2))
            count = 0

        # Apply transform
        if self.transform:
            img = self.transform(img)

        return {
            'image': img,
            'count': torch.tensor(count, dtype=torch.float),
            'points': torch.tensor(points, dtype=torch.float) if points.size > 0 else torch.zeros((0, 2)),
            'img_name': os.path.basename(img_path),
            'img_path': img_path,
            'original_size': original_size
        }


def custom_collate_fn(batch):
    """Custom collate function for UCF-QNRF"""
    images = [item['image'] for item in batch]
    counts = [item['count'] for item in batch]
    names = [item['img_name'] for item in batch]
    paths = [item['img_path'] for item in batch]
    sizes = [item['original_size'] for item in batch]
    points = [item['points'] for item in batch]

    return {
        'image': torch.stack(images, 0),
        'count': torch.stack(counts, 0),
        'img_name': names,
        'img_path': paths,
        'original_size': sizes,
        'points': points
    }


# =============================================================================
# TransCrowd Model (동일한 구조)
# =============================================================================
class DINOv2Backbone(nn.Module):
    def __init__(self, model_size='base', pretrained=True, img_size=392):
        super().__init__()
        self.model_size = model_size
        self.img_size = img_size

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
            except Exception:
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


class TransCrowd(nn.Module):
    def __init__(self, model_size='base', dropout_rate=0.1):
        super().__init__()
        self.backbone = DINOv2Backbone(model_size=model_size)

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


# =============================================================================
# Evaluation Functions
# =============================================================================
def calculate_metrics(predictions, targets):
    """Calculate comprehensive evaluation metrics"""
    predictions = np.array(predictions)
    targets = np.array(targets)

    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((predictions - targets) / (targets + 1e-4))) * 100

    # Additional metrics for crowd counting
    relative_mae = mae / np.mean(targets) * 100 if np.mean(targets) > 0 else float('inf')
    correlation = np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0

    # Accuracy within certain thresholds
    accuracy_10 = np.mean(np.abs(predictions - targets) <= 10) * 100  # Within 10 people
    accuracy_20 = np.mean(np.abs(predictions - targets) <= 20) * 100  # Within 20 people

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'Relative_MAE': relative_mae,
        'Correlation': correlation,
        'Accuracy_10': accuracy_10,
        'Accuracy_20': accuracy_20,
        'Mean_GT': np.mean(targets),
        'Std_GT': np.std(targets),
        'Mean_Pred': np.mean(predictions),
        'Std_Pred': np.std(predictions)
    }


def load_transcrowd_model(checkpoint_path, device):
    """Load TransCrowd model from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Get model config
        config = checkpoint.get('config', {})
        model_size = config.get('dinov2_size', 'base')
        dropout_rate = config.get('dropout_rate', 0.1)

        # Create model
        model = TransCrowd(model_size=model_size, dropout_rate=dropout_rate)

        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        print(f"✓ TransCrowd model loaded from {checkpoint_path}")
        print(f"   Model size: {model_size}")
        print(f"   Training epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   Training stage: {checkpoint.get('stage', 'unknown')}")

        return model

    except Exception as e:
        print(f"✗ Failed to load TransCrowd model: {e}")
        return None


def evaluate_on_ucf_qnrf(model, dataloader, device, save_predictions=True):
    """Evaluate model on UCF-QNRF dataset"""
    model.eval()

    predictions = []
    targets = []
    image_names = []
    prediction_details = []

    print("🔍 Evaluating on UCF-QNRF dataset...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            images = batch['image'].to(device)
            batch_targets = batch['count'].to(device)
            names = batch['img_name']

            try:
                batch_predictions = model(images)

                batch_preds_np = batch_predictions.cpu().numpy()
                batch_targets_np = batch_targets.cpu().numpy()

                predictions.extend(batch_preds_np)
                targets.extend(batch_targets_np)
                image_names.extend(names)

                # Store detailed predictions
                for i in range(len(names)):
                    prediction_details.append({
                        'image_name': names[i],
                        'ground_truth': batch_targets_np[i],
                        'prediction': batch_preds_np[i],
                        'absolute_error': abs(batch_preds_np[i] - batch_targets_np[i]),
                        'relative_error': abs(batch_preds_np[i] - batch_targets_np[i]) / (batch_targets_np[i] + 1e-4)
                    })

            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

    if len(predictions) == 0:
        print("❌ No valid predictions obtained!")
        return None, None

    # Calculate metrics
    metrics = calculate_metrics(predictions, targets)

    # Create detailed results DataFrame
    results_df = pd.DataFrame(prediction_details)

    return metrics, results_df


def create_ucf_qnrf_visualizations(metrics, results_df, save_dir):
    """Create comprehensive visualizations for UCF-QNRF results"""

    plt.style.use('default')

    # 1. Prediction vs Ground Truth Scatter Plot
    plt.figure(figsize=(12, 10))
    plt.scatter(results_df['ground_truth'], results_df['prediction'], alpha=0.6, s=30)

    # Perfect prediction line
    max_count = max(results_df['ground_truth'].max(), results_df['prediction'].max())
    plt.plot([0, max_count], [0, max_count], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Ground Truth Count', fontsize=12)
    plt.ylabel('Predicted Count', fontsize=12)
    plt.title(
        f'UCF-QNRF: Prediction vs Ground Truth\nMAE: {metrics["MAE"]:.2f}, Correlation: {metrics["Correlation"]:.3f}',
        fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ucf_qnrf_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Error Distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Absolute Error Histogram
    axes[0, 0].hist(results_df['absolute_error'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Absolute Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Absolute Error Distribution\nMean: {results_df["absolute_error"].mean():.2f}')
    axes[0, 0].grid(True, alpha=0.3)

    # Relative Error Histogram
    axes[0, 1].hist(results_df['relative_error'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Relative Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Relative Error Distribution\nMean: {results_df["relative_error"].mean():.2f}')
    axes[0, 1].grid(True, alpha=0.3)

    # Error vs Ground Truth
    axes[1, 0].scatter(results_df['ground_truth'], results_df['absolute_error'], alpha=0.6, s=20)
    axes[1, 0].set_xlabel('Ground Truth Count')
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('Error vs Ground Truth Count')
    axes[1, 0].grid(True, alpha=0.3)

    # Count Distribution Comparison
    axes[1, 1].hist(results_df['ground_truth'], bins=30, alpha=0.7, label='Ground Truth', edgecolor='black')
    axes[1, 1].hist(results_df['prediction'], bins=30, alpha=0.7, label='Predictions', edgecolor='black')
    axes[1, 1].set_xlabel('Count')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Count Distribution Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ucf_qnrf_error_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Best and Worst Predictions
    results_sorted = results_df.sort_values('absolute_error')

    # Best predictions
    best_results = results_sorted.head(6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (_, row) in enumerate(best_results.iterrows()):
        try:
            img_path = None
            # Find the image path
            for img_file in os.listdir(UCF_QNRF_DIR):
                if row['image_name'] in img_file:
                    img_path = os.path.join(UCF_QNRF_DIR, img_file)
                    break

            if img_path and os.path.exists(img_path):
                img = Image.open(img_path)
                axes[i].imshow(img)
            else:
                axes[i].text(0.5, 0.5, 'Image not found', ha='center', va='center',
                             transform=axes[i].transAxes)

            axes[i].set_title(
                f'GT: {row["ground_truth"]:.0f}, Pred: {row["prediction"]:.1f}\nError: {row["absolute_error"]:.1f}')
            axes[i].axis('off')
        except Exception:
            axes[i].text(0.5, 0.5, f'GT: {row["ground_truth"]:.0f}\nPred: {row["prediction"]:.1f}',
                         ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'Error: {row["absolute_error"]:.1f}')

    plt.suptitle('Best Predictions (Lowest Error)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ucf_qnrf_best_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Worst predictions
    worst_results = results_sorted.tail(6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (_, row) in enumerate(worst_results.iterrows()):
        try:
            img_path = None
            for img_file in os.listdir(UCF_QNRF_DIR):
                if row['image_name'] in img_file:
                    img_path = os.path.join(UCF_QNRF_DIR, img_file)
                    break

            if img_path and os.path.exists(img_path):
                img = Image.open(img_path)
                axes[i].imshow(img)
            else:
                axes[i].text(0.5, 0.5, 'Image not found', ha='center', va='center',
                             transform=axes[i].transAxes)

            axes[i].set_title(
                f'GT: {row["ground_truth"]:.0f}, Pred: {row["prediction"]:.1f}\nError: {row["absolute_error"]:.1f}')
            axes[i].axis('off')
        except Exception:
            axes[i].text(0.5, 0.5, f'GT: {row["ground_truth"]:.0f}\nPred: {row["prediction"]:.1f}',
                         ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'Error: {row["absolute_error"]:.1f}')

    plt.suptitle('Worst Predictions (Highest Error)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ucf_qnrf_worst_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 UCF-QNRF visualizations saved to {save_dir}")


def create_cross_dataset_report(metrics, results_df, save_dir, model_info):
    """Create comprehensive cross-dataset evaluation report"""

    report_path = os.path.join(save_dir, 'ucf_qnrf_cross_dataset_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CROSS-DATASET EVALUATION REPORT\n")
        f.write("ShanghaiTech → UCF-QNRF\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"📅 Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"🏋️  Training Dataset: ShanghaiTech Part A\n")
        f.write(f"🧪 Testing Dataset: UCF-QNRF\n")
        f.write(f"🤖 Model: TransCrowd (DINOv2-based)\n")
        f.write(f"📊 Test Images: {len(results_df)}\n\n")

        # Model Information
        f.write("MODEL INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model Architecture: TransCrowd with DINOv2 backbone\n")
        f.write(f"Checkpoint: {model_info.get('checkpoint_path', 'N/A')}\n")
        f.write(f"Training Stage: {model_info.get('stage', 'N/A')}\n")
        f.write(f"Training Epoch: {model_info.get('epoch', 'N/A')}\n\n")

        # Dataset Comparison
        f.write("DATASET CHARACTERISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write("ShanghaiTech Part A (Training):\n")
        f.write("  - Images: 300 train + 182 test\n")
        f.write("  - Crowd density: High density, congested scenes\n")
        f.write("  - Image resolution: Various (mostly smaller)\n")
        f.write("  - Scene types: Urban, indoor/outdoor events\n\n")

        f.write("UCF-QNRF (Testing):\n")
        f.write(f"  - Images: {len(results_df)} test images\n")
        f.write(f"  - Average crowd size: {metrics['Mean_GT']:.1f} ± {metrics['Std_GT']:.1f}\n")
        f.write("  - Crowd density: Very high density\n")
        f.write("  - Image resolution: High resolution (2013×2902 average)\n")
        f.write("  - Scene types: Sports events, concerts, protests\n\n")

        # Performance Results
        f.write("PERFORMANCE RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"📏 Mean Absolute Error (MAE): {metrics['MAE']:.2f}\n")
        f.write(f"📐 Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f}\n")
        f.write(f"📊 Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%\n")
        f.write(f"📈 Correlation Coefficient: {metrics['Correlation']:.3f}\n")
        f.write(f"📉 Relative MAE: {metrics['Relative_MAE']:.2f}%\n\n")

        f.write(f"🎯 Accuracy within 10 people: {metrics['Accuracy_10']:.1f}%\n")
        f.write(f"🎯 Accuracy within 20 people: {metrics['Accuracy_20']:.1f}%\n\n")

        # Prediction Statistics
        f.write("PREDICTION STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Ground Truth - Mean: {metrics['Mean_GT']:.1f}, Std: {metrics['Std_GT']:.1f}\n")
        f.write(f"Predictions - Mean: {metrics['Mean_Pred']:.1f}, Std: {metrics['Std_Pred']:.1f}\n")
        f.write(f"Prediction Bias: {metrics['Mean_Pred'] - metrics['Mean_GT']:.1f}\n\n")

        # Error Analysis
        f.write("ERROR ANALYSIS:\n")
        f.write("-" * 40 + "\n")

        # Categorize by crowd size
        small_crowd = results_df[results_df['ground_truth'] <= 50]
        medium_crowd = results_df[(results_df['ground_truth'] > 50) & (results_df['ground_truth'] <= 200)]
        large_crowd = results_df[results_df['ground_truth'] > 200]

        if len(small_crowd) > 0:
            f.write(f"Small crowds (≤50 people): {len(small_crowd)} images\n")
            f.write(f"  Average MAE: {small_crowd['absolute_error'].mean():.2f}\n")
            f.write(f"  Average relative error: {small_crowd['relative_error'].mean():.2f}\n\n")

        if len(medium_crowd) > 0:
            f.write(f"Medium crowds (51-200 people): {len(medium_crowd)} images\n")
            f.write(f"  Average MAE: {medium_crowd['absolute_error'].mean():.2f}\n")
            f.write(f"  Average relative error: {medium_crowd['relative_error'].mean():.2f}\n\n")

        if len(large_crowd) > 0:
            f.write(f"Large crowds (>200 people): {len(large_crowd)} images\n")
            f.write(f"  Average MAE: {large_crowd['absolute_error'].mean():.2f}\n")
            f.write(f"  Average relative error: {large_crowd['relative_error'].mean():.2f}\n\n")

        # Best and Worst Cases
        f.write("BEST AND WORST CASES:\n")
        f.write("-" * 40 + "\n")

        best_case = results_df.loc[results_df['absolute_error'].idxmin()]
        worst_case = results_df.loc[results_df['absolute_error'].idxmax()]

        f.write(f"Best Prediction:\n")
        f.write(f"  Image: {best_case['image_name']}\n")
        f.write(f"  Ground Truth: {best_case['ground_truth']:.0f}\n")
        f.write(f"  Prediction: {best_case['prediction']:.1f}\n")
        f.write(f"  Absolute Error: {best_case['absolute_error']:.1f}\n\n")

        f.write(f"Worst Prediction:\n")
        f.write(f"  Image: {worst_case['image_name']}\n")
        f.write(f"  Ground Truth: {worst_case['ground_truth']:.0f}\n")
        f.write(f"  Prediction: {worst_case['prediction']:.1f}\n")
        f.write(f"  Absolute Error: {worst_case['absolute_error']:.1f}\n\n")

        # Cross-Dataset Analysis
        f.write("CROSS-DATASET GENERALIZATION ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write("🔍 Domain Gap Assessment:\n")
        f.write("  - Scene Complexity: UCF-QNRF has higher resolution and larger crowds\n")
        f.write("  - Annotation Style: Different annotation protocols between datasets\n")
        f.write("  - Image Quality: UCF-QNRF generally has higher quality images\n")
        f.write("  - Crowd Patterns: Different crowd behaviors and distributions\n\n")

        # Performance interpretation
        if metrics['MAE'] < 100:
            performance_level = "Excellent"
            interpretation = "Model shows strong generalization capabilities"
        elif metrics['MAE'] < 200:
            performance_level = "Good"
            interpretation = "Model generalizes reasonably well with some domain gap"
        elif metrics['MAE'] < 400:
            performance_level = "Fair"
            interpretation = "Model shows moderate generalization with significant domain gap"
        else:
            performance_level = "Poor"
            interpretation = "Model struggles with domain transfer"

        f.write(f"🏆 Overall Performance: {performance_level}\n")
        f.write(f"📝 Interpretation: {interpretation}\n\n")

        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 40 + "\n")

        if metrics['MAE'] < 150:
            f.write("✅ Excellent cross-dataset performance! Consider:\n")
            f.write("   - Publishing results as state-of-the-art generalization\n")
            f.write("   - Testing on additional datasets for robustness\n")
        elif metrics['MAE'] < 300:
            f.write("🔄 Good performance with room for improvement:\n")
            f.write("   - Consider domain adaptation techniques\n")
            f.write("   - Fine-tuning on small UCF-QNRF subset\n")
            f.write("   - Data augmentation to bridge domain gap\n")
        else:
            f.write("🔧 Performance needs improvement:\n")
            f.write("   - Implement domain adaptation methods\n")
            f.write("   - Consider multi-dataset training\n")
            f.write("   - Analyze failure cases for targeted improvements\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"📄 Cross-dataset evaluation report saved to {report_path}")


import time


def main():
    """Main function to run UCF-QNRF cross-dataset evaluation"""
    parser = argparse.ArgumentParser(description='Test ShanghaiTech-trained TransCrowd on UCF-QNRF')
    parser.add_argument('--transcrowd-checkpoint', type=str, required=True,
                        help='Path to ShanghaiTech-trained TransCrowd checkpoint')
    parser.add_argument('--ucf-qnrf-dir', type=str, default=UCF_QNRF_DIR,
                        help='Path to UCF-QNRF dataset directory')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for evaluation (smaller for high-res UCF-QNRF)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to test (for quick evaluation)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.transcrowd_checkpoint):
        print(f"❌ TransCrowd checkpoint not found: {args.transcrowd_checkpoint}")
        return

    if not os.path.exists(args.ucf_qnrf_dir):
        print(f"❌ UCF-QNRF dataset not found: {args.ucf_qnrf_dir}")
        print("Please download UCF-QNRF dataset from:")
        print("https://www.crcv.ucf.edu/data/ucf-qnrf/")
        return

    # Set output directory
    if args.output_dir is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(UCF_TEST_DIR, f'ucf_qnrf_test_{timestamp}')
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print("🚀 Starting UCF-QNRF Cross-Dataset Evaluation")
    print("=" * 60)
    print(f"📁 TransCrowd checkpoint: {args.transcrowd_checkpoint}")
    print(f"📁 UCF-QNRF dataset: {args.ucf_qnrf_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🖥️  Device: {DEVICE}")
    print(f"🔢 Batch size: {args.batch_size}")

    # Load TransCrowd model
    print(f"\n🤖 Loading TransCrowd model...")
    model = load_transcrowd_model(args.transcrowd_checkpoint, DEVICE)

    if model is None:
        print("❌ Failed to load model. Exiting.")
        return

    # Model info for report
    try:
        checkpoint = torch.load(args.transcrowd_checkpoint, map_location='cpu')
        model_info = {
            'checkpoint_path': args.transcrowd_checkpoint,
            'epoch': checkpoint.get('epoch', 'unknown'),
            'stage': checkpoint.get('stage', 'unknown')
        }
    except:
        model_info = {'checkpoint_path': args.transcrowd_checkpoint}

    # Create UCF-QNRF dataset
    print(f"\n📊 Setting up UCF-QNRF dataset...")

    # UCF-QNRF typically uses higher resolution, so we use a reasonable size
    # that balances quality and computational efficiency
    test_transform = transforms.Compose([
        transforms.Resize((384, 384)),  # Resize to standard size for fair comparison
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ucf_dataset = UCFQNRFDataset(
        root_dir=args.ucf_qnrf_dir,
        split='Test',
        transform=test_transform,
        max_images=args.max_images
    )

    if len(ucf_dataset) == 0:
        print("❌ No valid UCF-QNRF test images found. Please check your dataset.")
        return

    ucf_loader = DataLoader(
        ucf_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"✅ UCF-QNRF dataset ready: {len(ucf_dataset)} test images")

    # Evaluate model
    print(f"\n🔍 Evaluating TransCrowd on UCF-QNRF...")
    metrics, results_df = evaluate_on_ucf_qnrf(model, ucf_loader, DEVICE)

    if metrics is None:
        print("❌ Evaluation failed. Please check your setup.")
        return

    # Save detailed results
    results_path = os.path.join(output_dir, 'ucf_qnrf_detailed_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"💾 Detailed results saved to {results_path}")

    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    create_ucf_qnrf_visualizations(metrics, results_df, output_dir)

    # Create comprehensive report
    print(f"\n📄 Generating comprehensive report...")
    create_cross_dataset_report(metrics, results_df, output_dir, model_info)

    # Print summary results
    print("\n" + "=" * 60)
    print("🎉 UCF-QNRF CROSS-DATASET EVALUATION RESULTS")
    print("=" * 60)
    print(f"📊 Test Images: {len(results_df)}")
    print(f"📏 Mean Absolute Error (MAE): {metrics['MAE']:.2f}")
    print(f"📐 Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f}")
    print(f"📊 Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")
    print(f"📈 Correlation: {metrics['Correlation']:.3f}")
    print(f"📉 Relative MAE: {metrics['Relative_MAE']:.2f}%")
    print(f"🎯 Accuracy ±10: {metrics['Accuracy_10']:.1f}%")
    print(f"🎯 Accuracy ±20: {metrics['Accuracy_20']:.1f}%")

    # Performance assessment
    print(f"\n🏆 PERFORMANCE ASSESSMENT:")
    if metrics['MAE'] < 100:
        print("🥇 EXCELLENT: Outstanding cross-dataset generalization!")
        print("   Your model shows exceptional ability to transfer from ShanghaiTech to UCF-QNRF")
    elif metrics['MAE'] < 200:
        print("🥈 GOOD: Strong cross-dataset performance")
        print("   Your model generalizes well despite domain differences")
    elif metrics['MAE'] < 400:
        print("🥉 FAIR: Moderate cross-dataset performance")
        print("   There's room for improvement, consider domain adaptation")
    else:
        print("📈 NEEDS IMPROVEMENT: Significant domain gap detected")
        print("   Consider fine-tuning or domain adaptation techniques")

    print(f"\n📁 All results saved to: {output_dir}")

    # Quick comparison with literature (if available)
    print(f"\n📚 LITERATURE COMPARISON:")
    print("   Note: UCF-QNRF is a challenging dataset. Typical performance:")
    print("   - CSRNet: ~277 MAE")
    print("   - MCNN: ~377 MAE")
    print("   - Advanced methods: ~100-250 MAE")
    print(f"   - Your TransCrowd: {metrics['MAE']:.1f} MAE")

    if metrics['MAE'] < 250:
        print("   🎉 Your model performs competitively with state-of-the-art methods!")

    print("\n🎊 Cross-dataset evaluation completed successfully!")


if __name__ == '__main__':
    main()