"""
Dataset class for ShanghaiTech crowd counting dataset.
"""

import os
import glob
import numpy as np
import scipy.io as sio
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter

from src.config import DATA_DIR, PART_A_DIR, PART_B_DIR, TRAIN_CONFIG
from torch.utils.data._utils.collate import default_collate


class ShanghaiTechDataset(Dataset):
    """
    Dataset class for ShanghaiTech crowd counting dataset.

    Args:
        part (str): Dataset part, either 'A' or 'B'.
        split (str): Dataset split, either 'train' or 'test'.
        transform (callable, optional): Transform to be applied on the image.
    """

    def __init__(self, part='A', split='train', transform=None):
        self.part = part
        self.split = split
        self.transform = transform

        # Set root directory based on part
        if part == 'A':
            self.root_dir = PART_A_DIR
        elif part == 'B':
            self.root_dir = PART_B_DIR
        else:
            raise ValueError("Part must be either 'A' or 'B'")

        possible_img_dirs = [
            os.path.join(self.root_dir, f'{split}_data', 'images'),  # Standard format
            os.path.join(self.root_dir, f'{split}', 'images'),  # Alternative 1
            os.path.join(self.root_dir, f'{split}_data', 'img'),  # Alternative 2
            os.path.join(self.root_dir, 'images', f'{split}'),  # Alternative 3
            os.path.join(self.root_dir, f'{split}'),  # Alternative 4
            os.path.join(self.root_dir, 'images')  # Alternative 5
        ]

        # Find the first directory that exists
        self.img_dir = None
        for dir_path in possible_img_dirs:
            if os.path.exists(dir_path):
                self.img_dir = dir_path
                print(f"Found images directory: {self.img_dir}")
                break

        if self.img_dir is None:
            print(f"WARNING: Could not find images directory for part {part}, split {split}!")
            print(f"Tried these paths: {possible_img_dirs}")
            self.img_dir = possible_img_dirs[0]  # Use the standard path as fallback

        # Similarly for ground truth
        possible_gt_dirs = [
            os.path.join(self.root_dir, f'{split}_data', 'ground-truth'),
            os.path.join(self.root_dir, f'{split}_data', 'ground_truth'),
            os.path.join(self.root_dir, f'{split}_data', 'groundtruth'),
            os.path.join(self.root_dir, f'{split}', 'ground-truth'),
            os.path.join(self.root_dir, f'{split}', 'ground_truth'),
            os.path.join(self.root_dir, f'{split}', 'groundtruth'),
            os.path.join(self.root_dir, 'ground-truth', f'{split}'),
            os.path.join(self.root_dir, 'ground_truth', f'{split}'),
            os.path.join(self.root_dir, 'groundtruth', f'{split}'),
            os.path.join(self.root_dir, 'ground-truth'),
            os.path.join(self.root_dir, 'ground_truth'),
            os.path.join(self.root_dir, 'groundtruth')
        ]

        self.gt_dir = None
        for dir_path in possible_gt_dirs:
            if os.path.exists(dir_path):
                self.gt_dir = dir_path
                print(f"Found ground truth directory: {self.gt_dir}")
                break

        if self.gt_dir is None:
            print(f"WARNING: Could not find ground truth directory for part {part}, split {split}!")
            print(f"Tried these paths: {possible_gt_dirs}")
            self.gt_dir = possible_gt_dirs[0]  # Use the standard path as fallback

        # Get all image files
        img_extensions = ['*.jpg', '*.jpeg', '*.png']
        self.img_paths = []

        for ext in img_extensions:
            self.img_paths.extend(glob.glob(os.path.join(self.img_dir, ext)))

        if not self.img_paths:
            print(f"WARNING: No images found for part {part}, split {split}!")
            print(f"Looked in: {self.img_dir}")
            print(f"Make sure the path exists and contains images.")
        else:
            print(f"Found {len(self.img_paths)} images for part {part}, split {split}")

        self.img_paths.sort()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        # Get ground truth file path
        img_name = os.path.basename(img_path)
        img_id = img_name.split('.')[0]

        # Try to find the ground truth file with different naming patterns
        gt_patterns = [
            f'GT_{img_id}.mat',
            f'{img_id}.mat',
            f'GT_{img_id.lower()}.mat',
            f'GT_{img_id.upper()}.mat',
            f'{img_id}_ann.mat',
            f'{img_id.lower()}_ann.mat',
            f'{img_id.upper()}_ann.mat'
        ]

        gt_path = None
        for pattern in gt_patterns:
            path = os.path.join(self.gt_dir, pattern)
            if os.path.exists(path):
                gt_path = path
                break

        # Load ground truth or use default values if not found
        if gt_path and os.path.exists(gt_path):
            try:
                gt_data = sio.loadmat(gt_path)

                # Try to get point annotations from the .mat file
                # Different datasets may store this information differently
                points = None
                if 'image_info' in gt_data:
                    # Format used in the original ShanghaiTech dataset
                    try:
                        points = gt_data['image_info'][0, 0]['location'][0, 0]
                    except (KeyError, IndexError):
                        print(f"Warning: Could not extract location data from 'image_info' in {gt_path}")
                elif 'annPoints' in gt_data:
                    # Format used in some other variants
                    points = gt_data['annPoints']
                elif 'points' in gt_data:
                    # Another common format
                    points = gt_data['points']

                if points is None:
                    print(f"Warning: Unknown ground truth format in {gt_path}")
                    print(f"Available keys: {list(gt_data.keys())}")
                    points = np.zeros((0, 2))

                count = points.shape[0]  # Total count
            except Exception as e:
                print(f"Error loading ground truth file {gt_path}: {str(e)}")
                points = np.zeros((0, 2))
                count = 0
        else:
            # If no ground truth file found
            if self.split == 'train':
                print(f"Warning: No ground truth file found for training image {img_name}")
            points = np.zeros((0, 2))
            count = 0

        # Apply transforms to image
        if self.transform:
            img = self.transform(img)

        return {
            'image': img,
            'count': torch.tensor(count, dtype=torch.float),
            'points': torch.tensor(points, dtype=torch.float) if points.size > 0 else torch.zeros((0, 2)),
            'path': img_path
        }


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized point annotations.

    Args:
        batch (list): List of samples from the dataset.

    Returns:
        dict: Collated batch with properly handled point annotations.
    """
    # Extract items that need special handling
    images = [item['image'] for item in batch]
    counts = [item['count'] for item in batch]
    paths = [item['path'] for item in batch]

    # Handle points separately - they have variable sizes
    points = [item['points'] for item in batch]

    # Create the output dictionary
    output = {
        'image': torch.stack(images, 0),
        'count': torch.stack(counts, 0),
        'path': paths,
        # For points, we're not stacking them since they have variable dimensions
        'points': points
    }

    return output


def create_density_map(points, shape, sigma=15):
    """
    Create density map from point annotations.

    Args:
        points (numpy.ndarray): Array of points (x, y coordinates).
        shape (tuple): Shape of the density map (height, width).
        sigma (float): Sigma for Gaussian kernel.

    Returns:
        numpy.ndarray: Density map.
    """
    density_map = np.zeros(shape, dtype=np.float32)

    # If there are no points, return empty density map
    if points.shape[0] == 0:
        return density_map

    # Create density map by placing Gaussian at each point
    for i in range(points.shape[0]):
        point_x, point_y = points[i]
        point_x, point_y = min(shape[1]-1, max(0, int(point_x))), min(shape[0]-1, max(0, int(point_y)))
        density_map[point_y, point_x] = 1

    # Apply Gaussian filter
    density_map = gaussian_filter(density_map, sigma=sigma, truncate=3.0/sigma)

    # Normalize density map to preserve count
    if density_map.sum() > 0:
        density_map = density_map / density_map.sum() * points.shape[0]

    return density_map


def get_transforms(img_size=384):
    """
    Get transforms for training and testing.

    Args:
        img_size (int): Image size for resizing.

    Returns:
        tuple: (train_transform, test_transform)
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform


def get_dataloaders(part='A', batch_size=16, num_workers=4, pin_memory=True, img_size=384):
    """
    Create train and test dataloaders for ShanghaiTech dataset.

    Args:
        part (str): Dataset part, either 'A' or 'B'.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to pin memory for faster data transfer to GPU.
        img_size (int): Image size for resizing.

    Returns:
        tuple: (train_dataloader, test_dataloader)
    """
    # Get transforms
    train_transform, test_transform = get_transforms(img_size)

    # Create datasets
    train_dataset = ShanghaiTechDataset(part=part, split='train', transform=train_transform)
    test_dataset = ShanghaiTechDataset(part=part, split='test', transform=test_transform)

    # Create dataloaders with custom collate function
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=custom_collate_fn  # Add custom collate function
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn  # Add custom collate function
    )

    return train_dataloader, test_dataloader