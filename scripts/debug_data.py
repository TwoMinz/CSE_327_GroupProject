"""
Debug script to check dataset structure and configuration.
"""

import os
import glob
import argparse
import scipy.io as sio
from PIL import Image

from src.config import DATA_DIR, PART_A_DIR, PART_B_DIR


def explore_directory(directory, max_depth=3, current_depth=0):
    """
    Recursively explore a directory structure.

    Args:
        directory (str): Directory path to explore.
        max_depth (int): Maximum recursion depth.
        current_depth (int): Current recursion depth.
    """
    if current_depth > max_depth:
        print(f"{'  ' * current_depth}[MAX DEPTH REACHED]")
        return

    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return

    print(f"{'  ' * current_depth}{os.path.basename(directory)}/")

    try:
        items = sorted(os.listdir(directory))
        dirs = []
        files = []

        for item in items:
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                dirs.append(item)
            else:
                files.append(item)

        # Print directories first
        for dir_name in dirs:
            explore_directory(os.path.join(directory, dir_name), max_depth, current_depth + 1)

        # Print files (limit to 5 files per directory)
        if files:
            num_files = len(files)
            shown_files = files[:5]

            for file_name in shown_files:
                print(f"{'  ' * (current_depth + 1)}{file_name}")

            if num_files > 5:
                print(f"{'  ' * (current_depth + 1)}... and {num_files - 5} more files")
    except PermissionError:
        print(f"{'  ' * (current_depth + 1)}[PERMISSION DENIED]")


def check_image_files(part='A'):
    """
    Check for image files in ShanghaiTech dataset.

    Args:
        part (str): Dataset part.
    """
    root_dir = PART_A_DIR if part == 'A' else PART_B_DIR

    # Check the specific structure mentioned
    train_images_dir = os.path.join(root_dir, 'train_data', 'images')
    test_images_dir = os.path.join(root_dir, 'test_data', 'images')

    print(f"\nSearching for image files in part {part}:")

    # Check train images
    train_images = []
    if os.path.exists(train_images_dir):
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            train_images.extend(glob.glob(os.path.join(train_images_dir, ext)))
        print(f"  ✓ Found {len(train_images)} train images in {train_images_dir}")
    else:
        print(f"  ✗ Train images directory not found: {train_images_dir}")

    # Check test images
    test_images = []
    if os.path.exists(test_images_dir):
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_images.extend(glob.glob(os.path.join(test_images_dir, ext)))
        print(f"  ✓ Found {len(test_images)} test images in {test_images_dir}")
    else:
        print(f"  ✗ Test images directory not found: {test_images_dir}")

    # If no images found, try alternative locations
    if len(train_images) == 0 and len(test_images) == 0:
        print("  Looking for images in alternative locations...")

        # Try different patterns
        patterns = [
            os.path.join(root_dir, 'images', '*.jpg'),
            os.path.join(root_dir, 'images', '*.png'),
            os.path.join(root_dir, '*.jpg'),
            os.path.join(root_dir, '*.png')
        ]

        for pattern in patterns:
            images = glob.glob(pattern)
            if images:
                print(f"  ✓ Found {len(images)} images with pattern: {pattern}")

    return train_images, test_images


def check_ground_truth_files(part='A'):
    """
    Check for ground truth files in ShanghaiTech dataset.

    Args:
        part (str): Dataset part.
    """
    root_dir = PART_A_DIR if part == 'A' else PART_B_DIR

    # Check the specific structure mentioned
    train_gt_dir = os.path.join(root_dir, 'train_data', 'groundtruth')
    test_gt_dir = os.path.join(root_dir, 'test_data', 'groundtruth')

    print(f"\nSearching for ground truth files in part {part}:")

    # Check train ground truth
    train_gt = []
    if os.path.exists(train_gt_dir):
        train_gt = glob.glob(os.path.join(train_gt_dir, '*.mat'))
        print(f"  ✓ Found {len(train_gt)} train ground truth files in {train_gt_dir}")
    else:
        print(f"  ✗ Train ground truth directory not found: {train_gt_dir}")

    # Check test ground truth
    test_gt = []
    if os.path.exists(test_gt_dir):
        test_gt = glob.glob(os.path.join(test_gt_dir, '*.mat'))
        print(f"  ✓ Found {len(test_gt)} test ground truth files in {test_gt_dir}")
    else:
        print(f"  ✗ Test ground truth directory not found: {test_gt_dir}")

    # If no ground truth files found, try alternative locations
    if len(train_gt) == 0 and len(test_gt) == 0:
        print("  Looking for ground truth files in alternative locations...")

        # Try different patterns
        patterns = [
            os.path.join(root_dir, 'ground-truth', '*.mat'),
            os.path.join(root_dir, 'ground_truth', '*.mat'),
            os.path.join(root_dir, 'groundtruth', '*.mat')
        ]

        for pattern in patterns:
            gt_files = glob.glob(pattern)
            if gt_files:
                print(f"  ✓ Found {len(gt_files)} ground truth files with pattern: {pattern}")

    # Check a sample ground truth file if available
    sample_gt = None
    if train_gt:
        sample_gt = train_gt[0]
    elif test_gt:
        sample_gt = test_gt[0]

    if sample_gt:
        print(f"\n  Checking ground truth format using file: {os.path.basename(sample_gt)}")
        try:
            gt_data = sio.loadmat(sample_gt)
            print(f"  Ground truth file keys: {list(gt_data.keys())}")

            # Try different known ground truth formats
            if 'image_info' in gt_data:
                try:
                    points = gt_data['image_info'][0, 0]['location'][0, 0]
                    print(f"  ✓ Found 'image_info' structure with {points.shape[0]} points")
                except (KeyError, IndexError):
                    print(f"  ✗ 'image_info' key exists but couldn't extract location data")
            elif 'annPoints' in gt_data:
                points = gt_data['annPoints']
                print(f"  ✓ Found 'annPoints' with {points.shape[0]} points")
            elif 'points' in gt_data:
                points = gt_data['points']
                print(f"  ✓ Found 'points' with {points.shape[0]} points")
            else:
                print(f"  ✗ Could not find standard point data keys")
                print(f"  Available keys: {list(gt_data.keys())}")
        except Exception as e:
            print(f"  ✗ Error reading ground truth file: {str(e)}")

    return train_gt, test_gt


def check_image_ground_truth_pairs(part='A'):
    """
    Check if image and ground truth files form valid pairs.

    Args:
        part (str): Dataset part.
    """
    root_dir = PART_A_DIR if part == 'A' else PART_B_DIR

    # Check for train and test data
    for split in ['train', 'test']:
        img_dir = os.path.join(root_dir, f'{split}_data', 'images')
        gt_dir = os.path.join(root_dir, f'{split}_data', 'groundtruth')

        if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
            print(f"\nCannot check pairs for {split}_data: missing images or ground truth directories.")
            continue

        # Find all image files
        images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            images.extend([f for f in os.listdir(img_dir) if f.lower().endswith(ext)])

        # Find all ground truth files
        gt_files = [f for f in os.listdir(gt_dir) if f.lower().endswith('.mat')]

        print(f"\nChecking image and ground truth pairs in part {part}, {split}_data:")
        print(f"  Found {len(images)} images in {img_dir}")
        print(f"  Found {len(gt_files)} ground truth files in {gt_dir}")

        # Check a few images for corresponding ground truth
        max_samples = min(5, len(images))
        for i in range(max_samples):
            img_name = images[i]
            img_id = img_name.split('.')[0]
            img_path = os.path.join(img_dir, img_name)

            print(f"\n  Image {i + 1}/{max_samples}: {img_name}")

            # Look for corresponding ground truth file with different possible naming patterns
            gt_patterns = [f'GT_{img_id}.mat', f'{img_id}.mat', f'GT_{img_id.lower()}.mat', f'GT_{img_id.upper()}.mat']
            gt_found = False

            for pattern in gt_patterns:
                gt_path = os.path.join(gt_dir, pattern)
                if os.path.exists(gt_path):
                    print(f"  ✓ Found matching ground truth: {pattern}")

                    try:
                        # Try to open image
                        img = Image.open(img_path)
                        print(f"  ✓ Image size: {img.size}")

                        # Try to read ground truth
                        gt_data = sio.loadmat(gt_path)

                        # Try to extract point locations
                        if 'image_info' in gt_data:
                            try:
                                points = gt_data['image_info'][0, 0]['location'][0, 0]
                                print(f"  ✓ Found {points.shape[0]} points in 'image_info'")
                            except (KeyError, IndexError):
                                print(f"  ✗ 'image_info' exists but couldn't extract location data")
                        elif 'annPoints' in gt_data:
                            points = gt_data['annPoints']
                            print(f"  ✓ Found {points.shape[0]} points in 'annPoints'")
                        elif 'points' in gt_data:
                            points = gt_data['points']
                            print(f"  ✓ Found {points.shape[0]} points in 'points'")
                        else:
                            print(f"  ✗ Could not find point data in ground truth file")
                            print(f"  Available keys: {list(gt_data.keys())}")

                    except Exception as e:
                        print(f"  ✗ Error: {str(e)}")

                    gt_found = True
                    break

            if not gt_found:
                print(f"  ✗ No matching ground truth found for {img_name}")
                print(f"  Looked for: {gt_patterns}")


def debug_data():
    """
    Debug the ShanghaiTech dataset structure.
    """
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"WARNING: Data directory does not exist: {DATA_DIR}")
        return

    print(f"Data directory: {DATA_DIR}")

    # Explore directory structure
    print("\nDirectory structure:")
    explore_directory(DATA_DIR, max_depth=4)

    # Check for image files
    for part in ['A', 'B']:
        check_image_files(part)
        check_ground_truth_files(part)
        check_image_ground_truth_pairs(part)

    print("\nData directory debug completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debug ShanghaiTech dataset structure')
    parser.add_argument('--dir', type=str, default=None, help='custom data directory')
    args = parser.parse_args()

    if args.dir:
        # Override data directory
        DATA_DIR = args.dir
        PART_A_DIR = os.path.join(DATA_DIR, 'part_A')
        PART_B_DIR = os.path.join(DATA_DIR, 'part_B')

    debug_data()