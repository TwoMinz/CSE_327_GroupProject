"""
Crowd Counting Dataset Downloader and Setup Script
Downloads and sets up various crowd counting datasets for cross-validation.
"""

import os
import requests
import zipfile
import tarfile
import shutil
from tqdm import tqdm
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Dataset download information
DATASETS = {
    'UCF_QNRF': {
        'name': 'UCF-QNRF',
        'description': 'UCF-QNRF Dataset - High resolution diverse scenes',
        'url': 'https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip',
        'extract_to': 'UCF-QNRF_ECCV18',
        'size': '~1.3GB',
        'format': 'zip',
        'structure': {
            'train_images': 'Train',
            'test_images': 'Test',
            'train_gt': 'Train',
            'test_gt': 'Test'
        }
    },
    'JHU_CROWD': {
        'name': 'JHU-CROWD++',
        'description': 'JHU-CROWD++ Dataset - Weather and lighting variations',
        'url': 'http://www.crowd-counting.com/download/jhu_crowd_v2.0.zip',
        'extract_to': 'jhu_crowd_v2.0',
        'size': '~2.2GB',
        'format': 'zip',
        'structure': {
            'train_images': 'train/images',
            'test_images': 'test/images',
            'train_gt': 'train/gt',
            'test_gt': 'test/gt'
        }
    },
    'NWPU_CROWD': {
        'name': 'NWPU-Crowd',
        'description': 'NWPU-Crowd Dataset - Large scale crowd scenes',
        'url': 'https://gjy3035.github.io/NWPU-Crowd-Sample-Code/NWPU-Crowd.zip',
        'extract_to': 'NWPU-Crowd',
        'size': '~3.9GB',
        'format': 'zip',
        'structure': {
            'train_images': 'images_train',
            'test_images': 'images_test',
            'train_gt': 'gt_train',
            'test_gt': 'gt_test'
        }
    },
    'FDST': {
        'name': 'FDST',
        'description': 'FDST Dataset - Fudan-ShanghaiTech',
        'url': 'https://github.com/sweetyy83/Lstn_fdst_dataset',
        'extract_to': 'FDST',
        'size': '~800MB',
        'format': 'manual',
        'note': 'Requires manual download from GitHub'
    }
}


def create_directories():
    """Create necessary directories."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Data directory: {DATA_DIR}")


def download_file(url, filename, desc):
    """Download file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(filename, 'wb') as file, tqdm(
                desc=desc,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def extract_archive(filepath, extract_to):
    """Extract zip or tar archive."""
    try:
        if filepath.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif filepath.endswith(('.tar.gz', '.tgz', '.tar')):
            with tarfile.open(filepath, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Unsupported archive format: {filepath}")
            return False
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def download_dataset(dataset_key):
    """Download and setup a specific dataset."""
    dataset_info = DATASETS[dataset_key]
    dataset_dir = os.path.join(DATA_DIR, dataset_info['extract_to'])

    print(f"\n{'=' * 60}")
    print(f"Setting up: {dataset_info['name']}")
    print(f"Description: {dataset_info['description']}")
    print(f"Size: {dataset_info['size']}")
    print(f"{'=' * 60}")

    # Check if already exists
    if os.path.exists(dataset_dir):
        print(f"✓ Dataset already exists at: {dataset_dir}")
        return True

    # Handle manual download datasets
    if dataset_info['format'] == 'manual':
        print(f"❗ Manual download required")
        print(f"Please download from: {dataset_info['url']}")
        print(f"Extract to: {dataset_dir}")
        if 'note' in dataset_info:
            print(f"Note: {dataset_info['note']}")
        return False

    # Download dataset
    filename = os.path.join(DATA_DIR, f"{dataset_key}.{dataset_info['format']}")

    print(f"Downloading from: {dataset_info['url']}")
    if not download_file(dataset_info['url'], filename, f"Downloading {dataset_info['name']}"):
        return False

    print(f"Extracting to: {dataset_dir}")
    if not extract_archive(filename, DATA_DIR):
        return False

    # Clean up archive
    os.remove(filename)
    print(f"✓ Cleaned up archive file")

    # Verify structure
    verify_dataset_structure(dataset_dir, dataset_info)

    print(f"✓ {dataset_info['name']} setup completed!")
    return True


def verify_dataset_structure(dataset_dir, dataset_info):
    """Verify the downloaded dataset has expected structure."""
    print(f"Verifying dataset structure...")

    structure = dataset_info.get('structure', {})
    for component, path in structure.items():
        full_path = os.path.join(dataset_dir, path)
        if os.path.exists(full_path):
            print(f"  ✓ Found {component}: {path}")
        else:
            print(f"  ⚠️ Missing {component}: {path}")


def list_available_datasets():
    """List all available datasets."""
    print("Available Crowd Counting Datasets:")
    print("=" * 50)

    for key, info in DATASETS.items():
        status = "✓ Downloaded" if os.path.exists(os.path.join(DATA_DIR, info['extract_to'])) else "✗ Not downloaded"
        print(f"{key:15} | {info['name']:20} | {info['size']:8} | {status}")


def check_current_datasets():
    """Check which datasets are currently available."""
    print("Current Dataset Status:")
    print("=" * 50)

    available_count = 0
    for key, info in DATASETS.items():
        dataset_path = os.path.join(DATA_DIR, info['extract_to'])
        if os.path.exists(dataset_path):
            print(f"✓ {info['name']:25} | Available at: {dataset_path}")
            available_count += 1
        else:
            print(f"✗ {info['name']:25} | Not found")

    print(f"\nTotal available datasets: {available_count}/{len(DATASETS)}")

    # Also check ShanghaiTech
    shanghai_path = os.path.join(DATA_DIR, 'ShanghaiTech')
    if os.path.exists(shanghai_path):
        print(f"✓ {'ShanghaiTech':25} | Available at: {shanghai_path}")
    else:
        print(f"✗ {'ShanghaiTech':25} | Not found")


def create_dataset_info_file():
    """Create a dataset information file."""
    info_path = os.path.join(DATA_DIR, 'datasets_info.txt')

    with open(info_path, 'w') as f:
        f.write("Crowd Counting Datasets Information\n")
        f.write("=" * 50 + "\n\n")

        for key, info in DATASETS.items():
            f.write(f"Dataset: {info['name']}\n")
            f.write(f"Key: {key}\n")
            f.write(f"Description: {info['description']}\n")
            f.write(f"Size: {info['size']}\n")
            f.write(f"URL: {info['url']}\n")

            dataset_path = os.path.join(DATA_DIR, info['extract_to'])
            status = "Available" if os.path.exists(dataset_path) else "Not downloaded"
            f.write(f"Status: {status}\n")
            f.write("-" * 30 + "\n")

    print(f"Dataset information saved to: {info_path}")


def setup_for_evaluation():
    """Quick setup for cross-dataset evaluation."""
    print("🎯 Setting up datasets for cross-dataset evaluation...")
    print("This will download the most commonly used datasets.")

    # Recommended datasets for evaluation
    recommended = ['UCF_QNRF', 'JHU_CROWD']

    for dataset_key in recommended:
        if dataset_key in DATASETS:
            download_dataset(dataset_key)
        else:
            print(f"⚠️ Dataset {dataset_key} not found in configuration")

    print("\n🎉 Setup completed!")
    print("You can now run cross-dataset evaluation with:")
    print("python cross_dataset_evaluation.py")


def main():
    parser = argparse.ArgumentParser(description='Download and setup crowd counting datasets')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--check', action='store_true', help='Check current dataset status')
    parser.add_argument('--download', type=str, choices=list(DATASETS.keys()),
                        help='Download specific dataset')
    parser.add_argument('--download-all', action='store_true', help='Download all datasets')
    parser.add_argument('--setup-eval', action='store_true',
                        help='Quick setup for cross-dataset evaluation')

    args = parser.parse_args()

    create_directories()

    if args.list:
        list_available_datasets()
    elif args.check:
        check_current_datasets()
    elif args.download:
        download_dataset(args.download)
    elif args.download_all:
        for dataset_key in DATASETS.keys():
            download_dataset(dataset_key)
    elif args.setup_eval:
        setup_for_evaluation()
    else:
        # Default: show status and options
        check_current_datasets()
        print(f"\n💡 Usage options:")
        print(f"  --setup-eval    : Quick setup for evaluation (recommended)")
        print(f"  --download UCF_QNRF : Download specific dataset")
        print(f"  --download-all  : Download all datasets")
        print(f"  --list         : List available datasets")

    create_dataset_info_file()


if __name__ == '__main__':
    main()