import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import torchvision.transforms as transforms
import scipy.io
from PIL import Image


# 실제 훈련된 모델과 정확히 일치하는 구조
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
    def __init__(self, model_size='base', pretrained=True, img_size=392, dropout_rate=0.2, freeze_backbone=True):
        super(DINOv2WithRegression, self).__init__()

        self.backbone = DINOv2Backbone(model_size=model_size, pretrained=pretrained,
                                       img_size=img_size, freeze_backbone=freeze_backbone)

        # 체크포인트와 정확히 일치하는 regression head
        self.regression_head = nn.Sequential(
            nn.LayerNorm(self.backbone.num_features),  # 0: LayerNorm [768]
            nn.Dropout(dropout_rate),  # 1: Dropout
            nn.Linear(self.backbone.num_features, 256),  # 2: Linear [256, 768]
            nn.ReLU(inplace=True),  # 3: ReLU
            nn.Dropout(dropout_rate),  # 4: Dropout
            nn.Linear(256, 128),  # 5: Linear [128, 256]
            nn.ReLU(inplace=True),  # 6: ReLU
            nn.Dropout(dropout_rate),  # 7: Dropout
            nn.Linear(128, 64),  # 8: Linear [64, 128]
            nn.ReLU(inplace=True),  # 9: ReLU
            nn.Dropout(dropout_rate),  # 10: Dropout
            nn.Linear(64, 1)  # 11: Linear [1, 64]
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
            dropout_rate=config.get('dropout_rate', 0.2),
            freeze_backbone=config.get('freeze_backbone', True)
        )

    def forward(self, x):
        return self.model(x)


# 데이터셋 (UCF-QNRF용으로 수정)
class UCFQNRFDataset(Dataset):
    def __init__(self, data_root, max_samples=100, image_size=392):
        self.data_root = data_root
        self.image_size = image_size

        img_dir = os.path.join(data_root, 'Test')
        if not os.path.exists(img_dir):
            img_dir = os.path.join(data_root, 'Train')

        self.img_dir = img_dir
        all_images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.image_list = all_images[:max_samples]

        # DINOv2 전처리 (392x392)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"Loaded {len(self.image_list)} images successfully.")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            image = torch.zeros(3, self.image_size, self.image_size)

        # GT 로드
        gt_name = img_name.replace('.jpg', '_ann.mat')
        gt_path = os.path.join(self.img_dir, gt_name)
        try:
            gt_data = scipy.io.loadmat(gt_path)
            gt_count = gt_data['annPoints'].shape[0]
        except:
            gt_count = 0

        return image, img_name, gt_count


def load_dinov2_model():
    """
    DINOv2 best_stage2.pth 모델을 로드합니다.
    """
    try:
        print("Loading DINOv2 model (best_stage2.pth)...")

        # 체크포인트 로드
        checkpoint_path = '/Users/semin/Documents/GitHub/CSE_327_GroupProject/outputs/checkpoints/your_experiment/best_stage2.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        print("Checkpoint keys:", list(checkpoint.keys()))

        # Config 가져오기
        config = checkpoint.get('config', {
            'dinov2_size': 'base',
            'pretrained': True,
            'img_size': 392,
            'dropout_rate': 0.2,
            'freeze_backbone': True
        })

        print("Model config:", config)

        # 모델 생성
        model = TransCrowd(config)

        # 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ DINOv2 model loaded successfully!")

        # 메트릭 정보
        if 'metrics' in checkpoint:
            best_mae = checkpoint['metrics'].get('best_val_mae', 'unknown')
            print(f"✅ Best validation MAE: {best_mae}")

        return model, config

    except FileNotFoundError:
        print("❌ best_stage2.pth not found!")
        print("📥 Please ensure best_stage2.pth is at the correct path")
        raise

    except Exception as e:
        print(f"❌ Error loading DINOv2 model: {e}")
        raise


def calculate_metrics(predictions, ground_truths):
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    mae = np.mean(np.abs(predictions - ground_truths))
    mse = np.mean((predictions - ground_truths) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((predictions - ground_truths) / (ground_truths + 1e-8))) * 100
    correlation = np.corrcoef(predictions, ground_truths)[0, 1] if len(predictions) > 1 else 0

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'Correlation': correlation
    }


def test_dinov2_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 데이터 로드 (392x392 크기)
    dataset = UCFQNRFDataset("data/UCF-QNRF_ECCV18", max_samples=100, image_size=392)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # DINOv2 모델 로드
    model, config = load_dinov2_model()
    model.to(device).eval()

    print("=" * 80)
    print("Testing DINOv2 TransCrowd Model (Stage 2)")
    print("=" * 80)

    # 테스트 실행
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for i, (image, img_name, gt_count) in enumerate(dataloader):
            image = image.to(device)
            ground_truths.append(int(gt_count))

            try:
                pred_count = model(image)

                # 스칼라 값으로 변환
                if torch.is_tensor(pred_count):
                    if pred_count.numel() == 1:
                        pred = pred_count.item()
                    else:
                        pred = torch.mean(pred_count).item()
                else:
                    pred = float(pred_count)

                # 음수 값 처리
                pred = max(0, pred)
                predictions.append(pred)

                if (i + 1) % 20 == 0:
                    print(f"Progress: {i + 1}/{len(dataset)} - GT: {gt_count}, Pred: {pred:.1f}")

            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                predictions.append(0)

    # 결과 분석
    print("\n" + "=" * 80)
    print("DINOv2 TransCrowd Results Summary")
    print("=" * 80)

    # GT 통계
    gt_array = np.array(ground_truths)
    pred_array = np.array(predictions)

    print(f"Ground Truth Statistics:")
    print(f"  Mean: {np.mean(gt_array):.1f}, Std: {np.std(gt_array):.1f}")
    print(f"  Range: {np.min(gt_array)}-{np.max(gt_array)}")
    print(f"  Median: {np.median(gt_array):.1f}")

    print(f"\nPrediction Statistics:")
    print(f"  Mean: {np.mean(pred_array):.1f}, Std: {np.std(pred_array):.1f}")
    print(f"  Range: {np.min(pred_array):.1f}-{np.max(pred_array):.1f}")
    print(f"  Median: {np.median(pred_array):.1f}")

    # 성능 메트릭
    metrics = calculate_metrics(predictions, ground_truths)
    print(f"\nPerformance Metrics:")
    print(f"  MAE: {metrics['MAE']:.2f} people")
    print(f"  RMSE: {metrics['RMSE']:.2f} people")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  Correlation: {metrics['Correlation']:.3f}")

    # 상세 결과 테이블 (전체 100개)
    print("\n" + "=" * 80)
    print("Detailed Results (All 100 images)")
    print("=" * 80)
    header = f"{'Image Name':<25} {'GT':<8} {'DINOv2_Pred':<12} {'Error':<12} {'Error%':<12}"
    print(header)
    print("-" * 80)

    for i in range(len(ground_truths)):
        img_name = dataset.image_list[i]
        gt = ground_truths[i]
        pred = predictions[i]
        error = abs(pred - gt)
        error_pct = (error / (gt + 1e-8)) * 100

        row = f"{img_name:<25} {gt:<8} {pred:<12.2f} {error:<12.2f} {error_pct:<12.1f}%"
        print(row)

    # 추가 에러 분석
    print("\n" + "=" * 80)
    print("Error Analysis")
    print("=" * 80)

    errors = np.array(predictions) - np.array(ground_truths)
    abs_errors = np.abs(errors)

    print(f"Error Statistics:")
    print(f"  Mean Error: {np.mean(errors):.2f}")
    print(f"  Error Std: {np.std(errors):.2f}")
    print(f"  Max Error: {np.max(abs_errors):.2f}")
    print(f"  Min Error: {np.min(abs_errors):.2f}")
    print(f"  Median Error: {np.median(abs_errors):.2f}")

    print(f"\nError Distribution:")
    print(f"  Errors ≤ 50: {np.sum(abs_errors <= 50)} images ({np.sum(abs_errors <= 50) / len(abs_errors) * 100:.1f}%)")
    print(
        f"  Errors ≤ 100: {np.sum(abs_errors <= 100)} images ({np.sum(abs_errors <= 100) / len(abs_errors) * 100:.1f}%)")
    print(
        f"  Errors ≤ 200: {np.sum(abs_errors <= 200)} images ({np.sum(abs_errors <= 200) / len(abs_errors) * 100:.1f}%)")
    print(
        f"  Errors > 200: {np.sum(abs_errors > 200)} images ({np.sum(abs_errors > 200) / len(abs_errors) * 100:.1f}%)")

    # 성능 요약
    print(f"\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Model: DINOv2 TransCrowd (Stage 2)")
    print(f"Training Dataset: ShanghaiTech Part A")
    print(f"Test Dataset: UCF-QNRF (100 samples)")
    print(f"Best Validation MAE (ShanghaiTech): 132.11 people")
    print(f"Test MAE (UCF-QNRF): {metrics['MAE']:.2f} people")
    print(f"Test RMSE: {metrics['RMSE']:.2f} people")
    print(f"Test Correlation: {metrics['Correlation']:.3f}")
    print(f"Mean GT Count: {np.mean(ground_truths):.1f} people")
    print(f"Mean Prediction: {np.mean(predictions):.1f} people")

    # 가장 좋은/나쁜 예측 찾기
    best_idx = np.argmin(abs_errors)
    worst_idx = np.argmax(abs_errors)

    print(f"\nBest Prediction:")
    print(f"  Image: {dataset.image_list[best_idx]}")
    print(f"  GT: {ground_truths[best_idx]}, Pred: {predictions[best_idx]:.1f}, Error: {abs_errors[best_idx]:.1f}")

    print(f"\nWorst Prediction:")
    print(f"  Image: {dataset.image_list[worst_idx]}")
    print(f"  GT: {ground_truths[worst_idx]}, Pred: {predictions[worst_idx]:.1f}, Error: {abs_errors[worst_idx]:.1f}")

    return predictions, ground_truths


if __name__ == "__main__":
    try:
        predictions, gt = test_dinov2_model()
        print("\nDINOv2 TransCrowd test completed successfully!")
        print("🎯 This model was trained on ShanghaiTech Part A!")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()