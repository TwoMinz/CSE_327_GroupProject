import torch
import os


def inspect_checkpoint():
    """체크포인트 구조를 분석합니다."""
    checkpoint_path = '/Users/semin/Documents/GitHub/CSE_327_GroupProject/outputs/checkpoints/your_experiment/best_stage2.pth'

    print("🔍 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print("\n📋 Top-level keys:")
    for key in checkpoint.keys():
        print(f"  - {key}")

    if 'model_state_dict' in checkpoint:
        model_keys = list(checkpoint['model_state_dict'].keys())
        print(f"\n🧠 Model state dict keys ({len(model_keys)} total):")
        print("First 10 keys:")
        for key in model_keys[:10]:
            print(f"  - {key}")

        print("\nLast 10 keys:")
        for key in model_keys[-10:]:
            print(f"  - {key}")

        # regression_head 관련 키들 찾기
        regression_keys = [key for key in model_keys if 'regression_head' in key]
        print(f"\n🎯 Regression head keys ({len(regression_keys)} total):")
        for key in regression_keys:
            param_shape = checkpoint['model_state_dict'][key].shape
            print(f"  - {key}: {param_shape}")

    if 'config' in checkpoint:
        print(f"\n⚙️ Config:")
        for key, value in checkpoint['config'].items():
            print(f"  - {key}: {value}")

    if 'metrics' in checkpoint:
        print(f"\n📊 Metrics:")
        for key, value in checkpoint['metrics'].items():
            print(f"  - {key}: {value}")


if __name__ == "__main__":
    inspect_checkpoint()