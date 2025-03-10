import os
import sys
import platform
import torch
import numpy as np
import cv2
import importlib.util
import subprocess
from tabulate import tabulate


def check_gpu():
    """Check GPU availability and details"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()

        gpu_info = []
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_total_memory = torch.cuda.get_device_properties(i).total_memory / (
                1024**3
            )  # Convert to GB
            gpu_info.append([i, gpu_name, f"{gpu_total_memory:.2f} GB"])

        print("\n===== GPU Information =====")
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {gpu_count}")

        headers = ["Index", "Name", "Memory"]
        print(tabulate(gpu_info, headers=headers, tablefmt="grid"))

        # Set environment variable for using GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print(f"Using GPU 0 by default. Set CUDA_VISIBLE_DEVICES to change.")

        return True, gpu_info
    else:
        print("\n===== GPU Information =====")
        print("CUDA Available: No")
        print("Warning: No GPU detected. Training will be extremely slow on CPU.")
        return False, None


def check_pytorch():
    """Check PyTorch installation"""
    try:
        import torch
        import torchvision

        print("\n===== PyTorch Information =====")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"TorchVision Version: {torchvision.__version__}")
        print(f"PyTorch Built with CUDA: {torch.version.cuda}")

        # Check if PyTorch can actually use CUDA
        if torch.cuda.is_available():
            # Create a small test tensor and move it to GPU
            try:
                x = torch.rand(10, 10).cuda()
                y = x + x
                del x, y
                print("PyTorch CUDA Test: Passed")
            except Exception as e:
                print(f"PyTorch CUDA Test: Failed - {str(e)}")

        return True
    except ImportError:
        print("\n===== PyTorch Information =====")
        print("PyTorch: Not installed or not in path")
        return False


def check_libraries():
    """Check necessary libraries"""
    libraries = [
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "PIL",
        "opencv-python",
        "albumentations",
        "timm",
        "tqdm",
        "pyyaml",
        "tensorboard",
        "h5py",
    ]

    results = []

    for lib in libraries:
        package_name = lib.split("-")[0] if "-" in lib else lib

        if package_name == "PIL":
            package_name = "pillow"

        if package_name == "opencv-python":
            package_name = "cv2"

        try:
            if package_name == "pillow":
                # Check PIL/Pillow special case
                from PIL import Image

                version = Image.__version__
            elif package_name == "cv2":
                # Check OpenCV special case
                version = cv2.__version__
            else:
                # General case
                spec = importlib.util.find_spec(package_name)
                if spec is None:
                    results.append([lib, "Not installed", "Error"])
                    continue

                module = importlib.import_module(package_name)
                if hasattr(module, "__version__"):
                    version = module.__version__
                elif hasattr(module, "version"):
                    version = module.version
                else:
                    version = "Unknown"

            results.append([lib, version, "OK"])
        except ImportError:
            results.append([lib, "Not installed", "Error"])
        except Exception as e:
            results.append([lib, "Error", str(e)])

    print("\n===== Required Libraries =====")
    headers = ["Library", "Version", "Status"]
    print(tabulate(results, headers=headers, tablefmt="grid"))

    missing = [row[0] for row in results if row[2] == "Error"]
    return len(missing) == 0, missing


def check_system():
    """Check system information"""
    print("\n===== System Information =====")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Processor: {platform.processor()}")

    # Check available RAM
    try:
        if platform.system() == "Windows":
            # Windows
            import psutil

            ram_gb = psutil.virtual_memory().total / (1024**3)
            print(f"RAM: {ram_gb:.2f} GB")
        elif platform.system() == "Linux":
            # Linux
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        ram_kb = int(line.split()[1])
                        ram_gb = ram_kb / (1024**2)
                        print(f"RAM: {ram_gb:.2f} GB")
                        break
        elif platform.system() == "Darwin":
            # macOS
            ram_bytes = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
            ram_gb = int(ram_bytes) / (1024**3)
            print(f"RAM: {ram_gb:.2f} GB")
    except Exception as e:
        print(f"RAM: Could not determine ({str(e)})")


def check_docker():
    """Check Docker installation"""
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            docker_version = result.stdout.strip()

            # Check Docker Compose
            compose_result = subprocess.run(
                ["docker-compose", "--version"], capture_output=True, text=True
            )
            if compose_result.returncode == 0:
                compose_version = compose_result.stdout.strip()
            else:
                compose_version = "Not installed or not in path"

            print("\n===== Docker Information =====")
            print(f"Docker: {docker_version}")
            print(f"Docker Compose: {compose_version}")
            return True
        else:
            print("\n===== Docker Information =====")
            print("Docker: Not installed or not in path")
            return False
    except FileNotFoundError:
        print("\n===== Docker Information =====")
        print("Docker: Not installed or not in path")
        return False


def print_summary(
    gpu_available, pytorch_ok, libraries_ok, missing_libraries, docker_ok
):
    """Print summary of checks"""
    print("\n===== Summary =====")

    if gpu_available and pytorch_ok and libraries_ok and docker_ok:
        print("✅ All checks passed! System is ready for training.")
    else:
        print("⚠️ Some checks failed. Please address the following issues:")

        if not gpu_available:
            print("  - No GPU detected. Training will be very slow on CPU.")

        if not pytorch_ok:
            print(
                "  - PyTorch installation issues. Please reinstall PyTorch with CUDA support."
            )

        if not libraries_ok:
            print(f"  - Missing libraries: {', '.join(missing_libraries)}")
            print(f"    Install with: pip install {' '.join(missing_libraries)}")

        if not docker_ok:
            print(
                "  - Docker not installed or not in path. Required for containerized training."
            )


def main():
    print("Running system compatibility check for CrowdViT...")

    # Check system information
    check_system()

    # Check GPU
    gpu_available, _ = check_gpu()

    # Check PyTorch
    pytorch_ok = check_pytorch()

    # Check libraries
    libraries_ok, missing_libraries = check_libraries()

    # Check Docker
    docker_ok = check_docker()

    # Print summary
    print_summary(gpu_available, pytorch_ok, libraries_ok, missing_libraries, docker_ok)


if __name__ == "__main__":
    main()
