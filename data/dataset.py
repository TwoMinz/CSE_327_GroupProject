import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm


class ShanghaiTechDataset(Dataset):
    """
    Dataset class for ShanghaiTech crowd counting dataset.
    Adapted for wait time prediction by mapping crowd density to estimated wait times.
    """

    def __init__(
        self, root_dir, part="A", phase="train", transform=None, target_size=(384, 384)
    ):
        """
        Args:
            root_dir (str): Root directory of the ShanghaiTech dataset
            part (str): 'A' or 'B' for different parts of the dataset
            phase (str): 'train' or 'test'
            transform: Optional transform to be applied on images
            target_size (tuple): Target size for resizing images
        """
        self.root_dir = root_dir
        self.part = part
        self.phase = phase
        self.transform = transform
        self.target_size = target_size

        # Define paths
        self.image_dir = os.path.join(
            root_dir, f"part_{part}", f"{phase}_data", "images"
        )
        self.ground_truth_dir = os.path.join(
            root_dir, f"part_{part}", f"{phase}_data", "ground-truth"
        )

        # Get image files
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))

        # Verify data exists
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

        print(f"Loaded {len(self.image_files)} images from {self.image_dir}")

        # Define default transforms if none provided
        if self.transform is None:
            if phase == "train":
                self.transform = A.Compose(
                    [
                        A.RandomResizedCrop(size=target_size, scale=(0.8, 1.0)),
                        A.HorizontalFlip(p=0.5),
                        A.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                        ),
                        A.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                        ToTensorV2(),
                    ]
                )
            else:
                self.transform = A.Compose(
                    [
                        A.Resize(height=target_size[0], width=target_size[1]),
                        A.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                        ToTensorV2(),
                    ]
                )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get corresponding ground truth file
        img_name = os.path.basename(img_path).replace(".jpg", "")
        gt_path = os.path.join(self.ground_truth_dir, f"GT_{img_name}.mat")

        # Load ground truth (density map)
        if os.path.exists(gt_path):
            gt_data = sio.loadmat(gt_path)
            density_map = gt_data["image_info"][0, 0][0, 0][0]  # Extract density map

            # Count total number of people
            num_people = np.sum(density_map)

            # Convert number of people to estimated wait time (minutes)
            # This is a simple heuristic - you may want to develop a more sophisticated mapping
            # based on your specific use case
            wait_time = self._estimate_wait_time(num_people)
        else:
            density_map = np.zeros((image.shape[0], image.shape[1]))
            num_people = 0
            wait_time = 0

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        sample = {
            "image": image,
            "num_people": torch.tensor(num_people, dtype=torch.float32),
            "wait_time": torch.tensor(wait_time, dtype=torch.float32),
            "image_path": img_path,
        }

        return sample

    def _estimate_wait_time(self, num_people):
        """
        Convert crowd count to estimated wait time.
        This is a simplified heuristic and should be calibrated for real-world scenarios.

        Args:
            num_people (float): Number of people in the image

        Returns:
            float: Estimated wait time in minutes
        """
        # Simple heuristic:
        # - 0-10 people: 0-5 minutes
        # - 10-30 people: 5-15 minutes
        # - 30-50 people: 15-25 minutes
        # - 50-100 people: 25-40 minutes
        # - 100+ people: 40+ minutes

        if num_people < 10:
            wait_time = num_people * 0.5  # 0.5 minutes per person for small crowds
        elif num_people < 30:
            wait_time = (
                5 + (num_people - 10) * 0.5
            )  # Base 5 minutes + 0.5 per additional person
        elif num_people < 50:
            wait_time = (
                15 + (num_people - 30) * 0.5
            )  # Base 15 minutes + 0.5 per additional person
        elif num_people < 100:
            wait_time = (
                25 + (num_people - 50) * 0.3
            )  # Base 25 minutes + 0.3 per additional person
        else:
            wait_time = (
                40 + (num_people - 100) * 0.2
            )  # Base 40 minutes + 0.2 per additional person

        return wait_time


def download_shanghaitech_dataset(download_dir="./data"):
    """
    Instructions to download and prepare the ShanghaiTech dataset
    """
    print("ShanghaiTech Dataset Download Instructions:")
    print(
        "1. Download the dataset from: https://www.kaggle.com/datasets/tthien/shanghaitech-with-people-density-map"
    )
    print(f"2. Extract the downloaded zip file to: {download_dir}/ShanghaiTech")
    print("3. Verify the dataset structure:")
    print("   - ShanghaiTech/")
    print("     |- part_A/")
    print("     |  |- train_data/")
    print("     |  |  |- images/")
    print("     |  |  |- ground-truth/")
    print("     |  |- test_data/")
    print("     |     |- images/")
    print("     |     |- ground-truth/")
    print("     |- part_B/")
    print("        |- train_data/")
    print("        |  |- images/")
    print("        |  |- ground-truth/")
    print("        |- test_data/")
    print("           |- images/")
    print("           |- ground-truth/")

    # Create directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    return


def create_dataloaders(
    root_dir, batch_size=16, num_workers=4, part="A", target_size=(384, 384)
):
    """
    Create train and validation dataloaders for ShanghaiTech dataset

    Args:
        root_dir (str): Root directory of the dataset
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        part (str): 'A' or 'B' for different parts of the dataset
        target_size (tuple): Target size for resizing images

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_dataset = ShanghaiTechDataset(
        root_dir=root_dir, part=part, phase="train", target_size=target_size
    )

    val_dataset = ShanghaiTechDataset(
        root_dir=root_dir,
        part=part,
        phase="test",  # Using test set as validation
        target_size=target_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def visualize_sample(dataset, idx=0):
    """
    Visualize a sample from the dataset

    Args:
        dataset: ShanghaiTech dataset
        idx (int): Index of the sample to visualize
    """
    sample = dataset[idx]

    # Convert tensor to numpy for visualization
    if isinstance(sample["image"], torch.Tensor):
        image = sample["image"].permute(1, 2, 0).numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
    else:
        image = sample["image"] / 255.0

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Image from {os.path.basename(sample['image_path'])}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.text(
        0.5,
        0.5,
        f"People count: {sample['num_people']:.1f}\nEstimated wait time: {sample['wait_time']:.1f} minutes",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=14,
        transform=plt.gca().transAxes,
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    dataset_root = "./data/ShanghaiTech"

    # Check if dataset exists, otherwise provide download instructions
    if not os.path.exists(dataset_root):
        download_shanghaitech_dataset()
    else:
        # Create dataset
        train_dataset = ShanghaiTechDataset(dataset_root, part="A", phase="train")

        # Visualize a sample
        visualize_sample(train_dataset)

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(dataset_root)

        print(f"Train loader: {len(train_loader)} batches")
        print(f"Val loader: {len(val_loader)} batches")
