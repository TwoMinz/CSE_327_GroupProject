import torch
import torchvision


def check_system():
    print("===== 시스템 정보 =====")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"TorchVision 버전: {torchvision.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU 개수: {torch.cuda.device_count()}")
        print(f"현재 GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )


if __name__ == "__main__":
    check_system()
