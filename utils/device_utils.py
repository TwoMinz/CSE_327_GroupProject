import torch
import platform
import os


def get_device(verbose=True):
    """
    Get the best available device for PyTorch operations.

    On Mac, attempts to use MPS (Metal Performance Shaders).
    On other systems, attempts to use CUDA.

    Args:
        verbose (bool): Whether to print device information

    Returns:
        torch.device: The best available device
    """
    # Check if we're on macOS
    is_mac = platform.system() == "Darwin"

    # Initialize device to CPU by default
    device = torch.device("cpu")
    device_name = "CPU"

    if is_mac:
        # Check for MPS (Metal Performance Shaders) availability on Mac
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            device_name = "MPS (Apple Silicon GPU)"
        else:
            device_name = "CPU (MPS not available)"
    else:
        # Check for CUDA on other platforms
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = f"CUDA ({torch.cuda.get_device_name(0)})"

    if verbose:
        print(f"Using device: {device_name}")

        # Print additional info
        if device.type == "cuda":
            print(f"CUDA Version: {torch.version.cuda}")
            print(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            )
        elif device.type == "mps":
            print("Using Apple Metal Performance Shaders (MPS) for GPU acceleration")

    return device


def move_model_to_device(model, device):
    """
    Move a PyTorch model to the specified device, handling
    any device-specific optimizations.

    Args:
        model (nn.Module): The PyTorch model
        device (torch.device): Target device

    Returns:
        nn.Module: The model on the target device
    """
    model = model.to(device)

    if device.type == "cuda":
        # CUDA-specific optimizations
        torch.backends.cudnn.benchmark = True
    elif device.type == "mps":
        # MPS-specific handling if needed
        pass

    return model


def get_dataloader_kwargs():
    """
    Get appropriate kwargs for DataLoader based on the platform.

    Returns:
        dict: Keyword arguments for DataLoader
    """
    kwargs = {}

    # For CUDA, we want to use pinned memory and enable non-blocking
    if torch.cuda.is_available():
        kwargs["pin_memory"] = True
        kwargs["pin_memory_device"] = "cuda"

    # MPS doesn't benefit from these settings

    return kwargs


if __name__ == "__main__":
    # Test the device detection
    device = get_device(verbose=True)
    print(f"Device type: {device.type}")

    # Check torch version
    print(f"PyTorch version: {torch.__version__}")

    # Create a test tensor
    test_tensor = torch.randn(1000, 1000)

    # Time CPU operation
    import time

    start = time.time()
    cpu_result = test_tensor @ test_tensor
    cpu_time = time.time() - start
    print(f"CPU matrix multiplication time: {cpu_time:.4f} seconds")

    # Time device operation if not CPU
    if device.type != "cpu":
        test_tensor = test_tensor.to(device)
        start = time.time()
        device_result = test_tensor @ test_tensor
        device_time = time.time() - start
        print(
            f"{device.type.upper()} matrix multiplication time: {device_time:.4f} seconds"
        )
        print(f"Speedup: {cpu_time / device_time:.2f}x")
