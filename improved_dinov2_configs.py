"""
Overfitting을 해결하기 위한 개선된 DINOv2 설정들
"""

import os
import copy

# 기본 설정 (현재 사용한 것)
BASE_CONFIG = {
    'backbone': 'dinov2',
    'img_size': 392,
    'dinov2_size': 'base',
    'pretrained': True,
    'freeze_backbone': True,
    'dropout_rate': 0.1,
}

BASE_TRAIN_CONFIG = {
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

# =============================================================================
# 실험 1: 강화된 정규화 (Regularization++)
# =============================================================================
EXPERIMENT_1_CONFIG = {
    'name': 'dinov2_regularized_v1',
    'description': '강화된 정규화로 overfitting 해결',
    'model_config': {
        **BASE_CONFIG,
        'dropout_rate': 0.3,  # 0.1 → 0.3 (3배 증가)
    },
    'train_config': {
        **BASE_TRAIN_CONFIG,
        'stage1_weight_decay': 5e-4,  # 1e-4 → 5e-4 (5배 증가)
        'stage2_weight_decay': 5e-4,  # Weight decay 강화
        'early_stopping_patience': 15,  # 더 빠른 조기 종료
        'stage2_epochs': 80,  # 훈련 epoch 줄이기
    }
}

# =============================================================================
# 실험 2: 작은 모델 + 강화된 Data Augmentation
# =============================================================================
EXPERIMENT_2_CONFIG = {
    'name': 'dinov2_small_augmented',
    'description': '작은 모델 + 강한 Data Augmentation',
    'model_config': {
        **BASE_CONFIG,
        'dinov2_size': 'small',  # base → small (파라미터 수 절반)
        'dropout_rate': 0.2,
    },
    'train_config': {
        **BASE_TRAIN_CONFIG,
        'batch_size': 24,  # 작은 모델로 배치 사이즈 증가
        'stage2_epochs': 100,
        'data_augmentation': 'strong',  # 강한 augmentation
    }
}

# =============================================================================
# 실험 3: 보수적 학습률 + 긴 Stage 1
# =============================================================================
EXPERIMENT_3_CONFIG = {
    'name': 'dinov2_conservative',
    'description': '보수적 학습률과 긴 Stage 1 훈련',
    'model_config': {
        **BASE_CONFIG,
        'dropout_rate': 0.25,
    },
    'train_config': {
        **BASE_TRAIN_CONFIG,
        'stage1_epochs': 50,  # 30 → 50 (더 긴 regression head 훈련)
        'stage2_epochs': 70,  # 짧은 fine-tuning
        'stage1_learning_rate': 5e-4,  # 1e-3 → 5e-4 (더 보수적)
        'stage2_learning_rate': 5e-6,  # 1e-5 → 5e-6 (더 보수적)
        'early_stopping_patience': 12,
    }
}

# =============================================================================
# 실험 4: 동결된 백본 (Frozen Backbone)
# =============================================================================
EXPERIMENT_4_CONFIG = {
    'name': 'dinov2_frozen_backbone',
    'description': 'Backbone 완전 동결로 overfitting 방지',
    'model_config': {
        **BASE_CONFIG,
        'freeze_backbone': True,  # Stage 2에서도 계속 동결
        'dropout_rate': 0.15,
    },
    'train_config': {
        **BASE_TRAIN_CONFIG,
        'stage1_epochs': 60,  # 더 긴 Stage 1
        'stage2_epochs': 40,  # 짧은 Stage 2 (백본 동결)
        'stage2_learning_rate': 5e-4,  # 더 높은 학습률 (regression head만)
        'freeze_backbone_stage2': True,  # Stage 2에서도 백본 동결
    }
}

# =============================================================================
# 실험 5: 균형잡힌 접근법
# =============================================================================
EXPERIMENT_5_CONFIG = {
    'name': 'dinov2_balanced',
    'description': '모든 기법을 적절히 조합한 균형잡힌 접근',
    'model_config': {
        **BASE_CONFIG,
        'dropout_rate': 0.2,
    },
    'train_config': {
        **BASE_TRAIN_CONFIG,
        'batch_size': 20,  # 약간 큰 배치
        'stage1_epochs': 40,
        'stage2_epochs': 60,
        'stage1_learning_rate': 7e-4,
        'stage2_learning_rate': 7e-6,
        'stage1_weight_decay': 3e-4,
        'stage2_weight_decay': 3e-4,
        'early_stopping_patience': 15,
        'grad_clip_norm': 0.3,  # 더 강한 gradient clipping
    }
}

# =============================================================================
# 실험 6: Label Smoothing + Mixup (고급 기법)
# =============================================================================
EXPERIMENT_6_CONFIG = {
    'name': 'dinov2_advanced_reg',
    'description': 'Label smoothing과 Mixup을 적용한 고급 정규화',
    'model_config': {
        **BASE_CONFIG,
        'dropout_rate': 0.15,
    },
    'train_config': {
        **BASE_TRAIN_CONFIG,
        'stage2_epochs': 80,
        'label_smoothing': 0.1,  # Label smoothing
        'mixup_alpha': 0.2,  # Mixup augmentation
        'early_stopping_patience': 18,
    }
}

EXPERIMENT_7_CONFIG = {
    'name': 'dinov2_max_performance',
    'description': '14GB VRAM 최대 활용 고성능 설정',
    'model_config': {
        'backbone': 'dinov2',
        'img_size': 448,  # 384 → 448 (더 큰 입력 해상도)
        'dinov2_size': 'large',  # base → large (더 큰 모델)
        'pretrained': True,
        'freeze_backbone': False,  # 전체 fine-tuning
        'dropout_rate': 0.15,  # 적절한 정규화
    },
    'train_config': {
        'part': 'A',
        'batch_size': 32,  # 16 → 32 (VRAM 여유로 배치 크기 증가)

        # 더 긴 훈련으로 성능 극대화
        'stage1_epochs': 60,  # 30 → 60 (더 긴 regression head 훈련)
        'stage2_epochs': 150,  # 120 → 150 (더 긴 fine-tuning)

        # 더 세밀한 학습률 설정
        'stage1_learning_rate': 3e-4,  # 더 낮은 시작 학습률
        'stage2_learning_rate': 1e-6,  # 매우 세밀한 fine-tuning

        # 강화된 정규화
        'stage1_weight_decay': 1e-3,
        'stage2_weight_decay': 1e-3,

        'lr_scheduler': 'cosine',
        'warmup_epochs': 10,  # 더 긴 warmup
        'early_stopping_patience': 25,  # 더 긴 patience
        'grad_clip_norm': 0.3,
        'seed': 42,
        'num_workers': 8,  # 더 많은 worker
        'pin_memory': True,
        'log_freq': 5,  # 더 자주 로그

        # 고급 기법들
        'data_augmentation': 'strong',
        'mixup_alpha': 0.1,
        'label_smoothing': 0.05,
    }
}

# =============================================================================
# 실험 8: 메모리 극한 도전 (Giant 모델)
# =============================================================================
EXPERIMENT_8_CONFIG = {
    'name': 'dinov2_giant_challenge',
    'description': '메모리 한계 도전 - Giant 모델',
    'model_config': {
        'backbone': 'dinov2',
        'img_size': 392,  # Giant는 메모리 많이 써서 해상도 조금 낮춤
        'dinov2_size': 'giant',  # 가장 큰 모델
        'pretrained': True,
        'freeze_backbone': True,  # Giant는 freeze해서 메모리 절약
        'dropout_rate': 0.2,
    },
    'train_config': {
        'part': 'A',
        'batch_size': 16,  # Giant 모델이라 배치 크기 조정

        'stage1_epochs': 80,  # 매우 긴 stage1 (백본 freeze)
        'stage2_epochs': 60,  # 짧은 stage2

        'stage1_learning_rate': 5e-4,
        'stage2_learning_rate': 5e-7,  # 매우 낮은 학습률

        'stage1_weight_decay': 5e-4,
        'stage2_weight_decay': 5e-4,

        'freeze_backbone_stage2': True,  # Stage2에서도 백본 동결

        'lr_scheduler': 'cosine',
        'warmup_epochs': 15,
        'early_stopping_patience': 20,
        'grad_clip_norm': 0.5,
        'seed': 42,
        'num_workers': 6,
        'pin_memory': True,
        'log_freq': 10,

        'data_augmentation': 'strong',
    }
}

# 모든 실험 설정 리스트
ALL_EXPERIMENTS = [
    EXPERIMENT_1_CONFIG,
    EXPERIMENT_2_CONFIG,
    EXPERIMENT_3_CONFIG,
    EXPERIMENT_4_CONFIG,
    EXPERIMENT_5_CONFIG,
    EXPERIMENT_6_CONFIG,
    EXPERIMENT_7_CONFIG,
    EXPERIMENT_8_CONFIG
]


def print_experiment_summary():
    """모든 실험 설정 요약 출력"""
    print("🧪 OVERFITTING 해결을 위한 실험 설정들")
    print("=" * 80)

    for i, exp in enumerate(ALL_EXPERIMENTS, 1):
        print(f"\n실험 {i}: {exp['name']}")
        print(f"설명: {exp['description']}")

        print("주요 변경사항:")
        model_changes = []
        train_changes = []

        # 모델 설정 변경사항
        for key, value in exp['model_config'].items():
            if key in BASE_CONFIG and BASE_CONFIG[key] != value:
                model_changes.append(f"  {key}: {BASE_CONFIG[key]} → {value}")

        # 훈련 설정 변경사항
        for key, value in exp['train_config'].items():
            if key in BASE_TRAIN_CONFIG and BASE_TRAIN_CONFIG[key] != value:
                train_changes.append(f"  {key}: {BASE_TRAIN_CONFIG[key]} → {value}")
            elif key not in BASE_TRAIN_CONFIG:
                train_changes.append(f"  {key}: {value} (새로 추가)")

        if model_changes:
            print("📊 모델 설정:")
            for change in model_changes:
                print(change)

        if train_changes:
            print("🏋️ 훈련 설정:")
            for change in train_changes:
                print(change)

        print(f"💻 실행 명령어:")
        print(f"python train_dinov2_improved.py --experiment {i}")


def get_experiment_config(experiment_id):
    """실험 ID로 설정 가져오기"""
    if 1 <= experiment_id <= len(ALL_EXPERIMENTS):
        return ALL_EXPERIMENTS[experiment_id - 1]
    else:
        raise ValueError(f"Invalid experiment ID: {experiment_id}. Choose 1-{len(ALL_EXPERIMENTS)}")


# 권장 실험 순서
RECOMMENDED_ORDER = [
    (1, "강화된 정규화", "가장 간단하고 효과적인 방법"),
    (3, "보수적 학습률", "안정적이고 신뢰할 수 있는 방법"),
    (5, "균형잡힌 접근", "여러 기법을 조합한 종합적 접근"),
    (2, "작은 모델", "빠른 실험과 비교용"),
]


def print_recommendations():
    """권장 실험 순서 출력"""
    print("\n🎯 권장 실험 순서:")
    print("=" * 50)

    for exp_id, name, reason in RECOMMENDED_ORDER:
        print(f"{exp_id}. {name}")
        print(f"   이유: {reason}")
        print(f"   실행: python train_dinov2_improved.py --experiment {exp_id}")
        print()


if __name__ == '__main__':
    print_experiment_summary()
    print_recommendations()