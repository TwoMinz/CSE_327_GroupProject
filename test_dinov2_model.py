"""
훈련된 DINOv2 모델 테스트 및 평가 스크립트
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import glob
import scipy.io as sio
from torch.utils.data import DataLoader, Dataset
import seaborn as sns

# Configuration (train_dinov2_standalone.py와 동일)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'ShanghaiTech')
PART_A_DIR = os.path.join(DATA_DIR, 'part_A')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')


def mean_absolute_error(pred, target):
    """Calculate mean absolute error."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    return np.mean(np.abs(pred - target))


def mean_squared_error(pred, target):
    """Calculate mean squared error."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    return np.mean((pred - target) ** 2)


def mean_absolute_percentage_error(pred, target, epsilon=1e-4):
    """Calculate mean absolute percentage error."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    return np.mean(np.abs((pred - target) / (target + epsilon))) * 100


# Dataset class (동일)
class ShanghaiTechDataset(Dataset):
    def __init__(self, part='A', split='test', transform=None):
        self.part = part
        self.split = split
        self.transform = transform

        if part == 'A':
            self.root_dir = PART_A_DIR
        else:
            raise ValueError("Only part A is configured")

        # Find directories
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

        print(f"Found {len(self.img_paths)} images for testing")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        # Get ground truth
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
            'path': img_path,
            'img_name': img_name
        }


def custom_collate_fn(batch):
    """Custom collate function."""
    images = [item['image'] for item in batch]
    counts = [item['count'] for item in batch]
    paths = [item['path'] for item in batch]
    names = [item['img_name'] for item in batch]
    points = [item['points'] for item in batch]

    return {
        'image': torch.stack(images, 0),
        'count': torch.stack(counts, 0),
        'path': paths,
        'img_name': names,
        'points': points
    }


def get_test_transforms(img_size=392):
    """Get test transforms"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# Model classes (동일한 구조)
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
                print(f"✓ Loaded DINOv2 {model_size} for testing")
            except Exception as e:
                print(f"Using fallback model for testing: {e}")
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
        except Exception:
            self.use_fallback = True
            self.backbone = self._create_basic_transformer().to(x.device)
            return self.backbone(x)


class DINOv2WithRegression(nn.Module):
    def __init__(self, model_size='base', pretrained=True, img_size=392, dropout_rate=0.1, freeze_backbone=False):
        super(DINOv2WithRegression, self).__init__()

        self.backbone = DINOv2Backbone(model_size=model_size, pretrained=pretrained,
                                       img_size=img_size, freeze_backbone=freeze_backbone)

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

        self.model = DINOv2WithRegression(
            model_size=config.get('dinov2_size', 'base'),
            pretrained=config.get('pretrained', True),
            img_size=config.get('img_size', 392),
            dropout_rate=config.get('dropout_rate', 0.1),
            freeze_backbone=config.get('freeze_backbone', False)
        )

    def forward(self, x):
        return self.model(x)


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    config = checkpoint.get('config', {
        'dinov2_size': 'base',
        'pretrained': True,
        'img_size': 392,
        'dropout_rate': 0.1,
        'freeze_backbone': False
    })

    # Create model
    model = TransCrowd(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"✓ Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"✓ Checkpoint stage: {checkpoint.get('stage', 'unknown')}")
    if 'metrics' in checkpoint:
        best_mae = checkpoint['metrics'].get('best_val_mae', 'unknown')
        print(f"✓ Best validation MAE: {best_mae}")

    return model, config


def evaluate_model(model, dataloader, device):
    """Comprehensive model evaluation"""
    model.eval()

    all_predictions = []
    all_targets = []
    all_names = []
    all_paths = []

    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            targets = batch['count'].to(device)

            predictions = model(images)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_names.extend(batch['img_name'])
            all_paths.extend(batch['path'])

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate metrics
    mae = mean_absolute_error(all_predictions, all_targets)
    mse = mean_squared_error(all_predictions, all_targets)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(all_predictions, all_targets)

    # Additional metrics
    correlation = np.corrcoef(all_predictions, all_targets)[0, 1]

    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'correlation': correlation,
        'num_samples': len(all_predictions)
    }

    # Create detailed results
    results_df = pd.DataFrame({
        'image_name': all_names,
        'image_path': all_paths,
        'ground_truth': all_targets,
        'prediction': all_predictions,
        'absolute_error': np.abs(all_predictions - all_targets),
        'relative_error': np.abs(all_predictions - all_targets) / (all_targets + 1e-4)
    })

    return metrics, results_df


def visualize_results(metrics, results_df, save_dir):
    """Create comprehensive visualizations"""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Metrics summary
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Scatter plot: Prediction vs Ground Truth
    axes[0, 0].scatter(results_df['ground_truth'], results_df['prediction'], alpha=0.6)
    axes[0, 0].plot([0, results_df['ground_truth'].max()], [0, results_df['ground_truth'].max()], 'r--',
                    label='Perfect prediction')
    axes[0, 0].set_xlabel('Ground Truth Count')
    axes[0, 0].set_ylabel('Predicted Count')
    axes[0, 0].set_title(f'Prediction vs Ground Truth\nCorrelation: {metrics["correlation"]:.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Error distribution
    axes[0, 1].hist(results_df['absolute_error'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Absolute Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Error Distribution\nMAE: {metrics["mae"]:.2f}')
    axes[0, 1].grid(True, alpha=0.3)

    # Relative error distribution
    axes[1, 0].hist(results_df['relative_error'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Relative Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Relative Error Distribution\nMAPE: {metrics["mape"]:.2f}%')
    axes[1, 0].grid(True, alpha=0.3)

    # Metrics bar chart
    metric_names = ['MAE', 'MSE', 'RMSE', 'MAPE']
    metric_values = [metrics['mae'], metrics['mse'], metrics['rmse'], metrics['mape']]
    axes[1, 1].bar(metric_names, metric_values)
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Evaluation Metrics')
    axes[1, 1].grid(True, alpha=0.3)

    # Add values on bars
    for i, v in enumerate(metric_values):
        axes[1, 1].text(i, v + max(metric_values) * 0.01, f'{v:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Best and worst predictions
    results_sorted = results_df.sort_values('absolute_error')

    # Best predictions (lowest error)
    best_results = results_sorted.head(6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (_, row) in enumerate(best_results.iterrows()):
        try:
            img = Image.open(row['image_path'])
            axes[i].imshow(img)
            axes[i].set_title(
                f'GT: {row["ground_truth"]:.0f}, Pred: {row["prediction"]:.1f}\nError: {row["absolute_error"]:.1f}')
            axes[i].axis('off')
        except Exception:
            axes[i].text(0.5, 0.5, 'Image not found', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'GT: {row["ground_truth"]:.0f}, Pred: {row["prediction"]:.1f}')

    plt.suptitle('Best Predictions (Lowest Error)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'best_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Worst predictions (highest error)
    worst_results = results_sorted.tail(6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (_, row) in enumerate(worst_results.iterrows()):
        try:
            img = Image.open(row['image_path'])
            axes[i].imshow(img)
            axes[i].set_title(
                f'GT: {row["ground_truth"]:.0f}, Pred: {row["prediction"]:.1f}\nError: {row["absolute_error"]:.1f}')
            axes[i].axis('off')
        except Exception:
            axes[i].text(0.5, 0.5, 'Image not found', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'GT: {row["ground_truth"]:.0f}, Pred: {row["prediction"]:.1f}')

    plt.suptitle('Worst Predictions (Highest Error)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'worst_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualizations saved to {save_dir}")


def test_single_image(model, image_path, transform, device):
    """Test model on a single image"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        prediction = model(img_tensor).item()

    return prediction


def main():
    parser = argparse.ArgumentParser(description='Test trained DINOv2 model')
    parser.add_argument('--experiment', type=str, required=True, help='experiment name (folder name)')
    parser.add_argument('--stage', type=str, default='stage2', choices=['stage1', 'stage2'],
                        help='which stage checkpoint to use')
    parser.add_argument('--checkpoint', type=str, default='best', choices=['best', 'latest'],
                        help='which checkpoint to use')
    parser.add_argument('--single-image', type=str, default=None, help='path to single image for testing')
    parser.add_argument('--save-results', action='store_true', help='save detailed results to CSV')

    args = parser.parse_args()

    print(f"Testing DINOv2 model: {args.experiment}")
    print(f"Using device: {DEVICE}")

    # Build checkpoint path
    checkpoint_path = os.path.join(CHECKPOINT_DIR, args.experiment, f'{args.checkpoint}_{args.stage}.pth')

    # Load model
    model, config = load_model(checkpoint_path, DEVICE)

    # Create output directory
    test_output_dir = os.path.join(OUTPUT_DIR, 'test_results', args.experiment)
    os.makedirs(test_output_dir, exist_ok=True)

    if args.single_image:
        # Test single image
        print(f"\nTesting single image: {args.single_image}")
        transform = get_test_transforms(config.get('img_size', 392))
        prediction = test_single_image(model, args.single_image, transform, DEVICE)
        print(f"Predicted count: {prediction:.2f}")

        # Visualize
        img = Image.open(args.single_image)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f'Predicted Count: {prediction:.2f}')
        plt.axis('off')
        plt.savefig(os.path.join(test_output_dir, 'single_image_prediction.png'), dpi=300, bbox_inches='tight')
        plt.show()

    else:
        # Test on full test dataset
        print(f"\nTesting on full test dataset...")

        # Create test dataset
        transform = get_test_transforms(config.get('img_size', 392))
        test_dataset = ShanghaiTechDataset(part='A', split='test', transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                                 num_workers=4, collate_fn=custom_collate_fn)

        # Evaluate
        metrics, results_df = evaluate_model(model, test_loader, DEVICE)

        # Print results
        print(f"\n{'=' * 50}")
        print("TEST RESULTS")
        print(f"{'=' * 50}")
        print(f"Number of test images: {metrics['num_samples']}")
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.2f}")
        print(f"Mean Squared Error (MSE): {metrics['mse']:.2f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
        print(f"Correlation: {metrics['correlation']:.3f}")
        print(f"{'=' * 50}")

        # Create visualizations
        visualize_results(metrics, results_df, test_output_dir)

        # Save detailed results
        if args.save_results:
            results_path = os.path.join(test_output_dir, 'detailed_results.csv')
            results_df.to_csv(results_path, index=False)
            print(f"✓ Detailed results saved to: {results_path}")

        # Save metrics summary
        metrics_path = os.path.join(test_output_dir, 'metrics_summary.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"DINOv2 Model Test Results\n")
            f.write(f"Experiment: {args.experiment}\n")
            f.write(f"Checkpoint: {args.checkpoint}_{args.stage}.pth\n")
            f.write(f"Number of test images: {metrics['num_samples']}\n")
            f.write(f"Mean Absolute Error (MAE): {metrics['mae']:.2f}\n")
            f.write(f"Mean Squared Error (MSE): {metrics['mse']:.2f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.2f}\n")
            f.write(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%\n")
            f.write(f"Correlation: {metrics['correlation']:.3f}\n")

        print(f"✓ Test completed! Results saved to: {test_output_dir}")


if __name__ == '__main__':
    main()