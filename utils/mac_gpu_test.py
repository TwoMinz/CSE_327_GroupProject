import sys
import os
import torch
import time
import platform
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import device utils
from utils.device_utils import get_device


def test_gpu_performance():
    """Test GPU performance by running matrix multiplication operations"""
    print("\n===== GPU Performance Test =====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Platform: {platform.platform()}")

    # Get device
    device = get_device(verbose=True)

    # Test matrix sizes
    sizes = [1000, 2000, 3000, 4000, 5000]
    cpu_times = []
    device_times = []

    for size in sizes:
        print(f"\nTesting {size}x{size} matrix multiplication:")

        # Create random matrices
        a = torch.randn(size, size)
        b = torch.randn(size, size)

        # Time CPU operation
        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.time()
        _ = torch.matmul(a, b)
        torch.cuda.synchronize() if device.type == "cuda" else None
        cpu_time = time.time() - start
        cpu_times.append(cpu_time)
        print(f"  CPU time: {cpu_time:.4f} seconds")

        # Time device operation if not CPU
        if device.type != "cpu":
            a_dev = a.to(device)
            b_dev = b.to(device)

            # Warmup
            _ = torch.matmul(a_dev, b_dev)

            torch.cuda.synchronize() if device.type == "cuda" else None
            start = time.time()
            _ = torch.matmul(a_dev, b_dev)
            torch.cuda.synchronize() if device.type == "cuda" else None
            device_time = time.time() - start
            device_times.append(device_time)
            print(f"  {device.type.upper()} time: {device_time:.4f} seconds")
            print(f"  Speedup: {cpu_time / device_time:.2f}x")

    # Plot results if we have device times
    if device_times:
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, cpu_times, "o-", label="CPU")
        plt.plot(sizes, device_times, "o-", label=device.type.upper())
        plt.xlabel("Matrix Size")
        plt.ylabel("Time (seconds)")
        plt.title("Matrix Multiplication Performance")
        plt.grid(True)
        plt.legend()

        # Save plot
        os.makedirs("./outputs", exist_ok=True)
        plt.savefig(f"./outputs/gpu_performance_{device.type}.png")
        plt.close()

        print(
            f"\nPerformance plot saved to ./outputs/gpu_performance_{device.type}.png"
        )


def test_model_training():
    """Test model training speed with a tiny model"""
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    print("\n===== Model Training Test =====")

    # Get device
    device = get_device(verbose=True)

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 32 * 8 * 8)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Create a small dataset
    batch_size = 64
    x = torch.randn(batch_size * 10, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size * 10,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model and optimizer
    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for a few epochs
    num_epochs = 5

    print(f"\nTraining for {num_epochs} epochs with batch size {batch_size}...")

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds")
    print(f"Average time per epoch: {total_time / num_epochs:.2f} seconds")


if __name__ == "__main__":
    # Run GPU tests
    test_gpu_performance()
    test_model_training()
