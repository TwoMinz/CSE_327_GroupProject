"""
Data augmentation transforms for crowd counting.
"""

import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms


class RandomHorizontalFlipWithPoints:
    """
    Random horizontal flip with point annotations.

    Args:
        p (float): Probability of flipping.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, points):
        """
        Args:
            img (PIL.Image): Image to be flipped.
            points (numpy.ndarray): Point annotations.

        Returns:
            tuple: (flipped_img, flipped_points)
        """
        if random.random() < self.p:
            width = img.width
            img = F.hflip(img)
            if points.shape[0] > 0:
                points[:, 0] = width - points[:, 0]
        return img, points


class RandomVerticalFlipWithPoints:
    """
    Random vertical flip with point annotations.

    Args:
        p (float): Probability of flipping.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, points):
        """
        Args:
            img (PIL.Image): Image to be flipped.
            points (numpy.ndarray): Point annotations.

        Returns:
            tuple: (flipped_img, flipped_points)
        """
        if random.random() < self.p:
            height = img.height
            img = F.vflip(img)
            if points.shape[0] > 0:
                points[:, 1] = height - points[:, 1]
        return img, points


class RandomCropWithPoints:
    """
    Random crop with point annotations.

    Args:
        size (tuple): Size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, img, points):
        """
        Args:
            img (PIL.Image): Image to be cropped.
            points (numpy.ndarray): Point annotations.

        Returns:
            tuple: (cropped_img, cropped_points)
        """
        width, height = img.size
        new_width, new_height = self.size

        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)

        img = F.crop(img, top, left, new_height, new_width)

        # Adjust points
        if points.shape[0] > 0:
            # Crop points
            mask = (
                    (points[:, 0] >= left) &
                    (points[:, 0] < left + new_width) &
                    (points[:, 1] >= top) &
                    (points[:, 1] < top + new_height)
            )
            points = points[mask]

            # Adjust coordinates
            points[:, 0] = points[:, 0] - left
            points[:, 1] = points[:, 1] - top

        return img, points


class Compose:
    """
    Compose several transforms together.

    Args:
        transforms (list): List of transforms.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, points):
        """
        Args:
            img (PIL.Image): Image to be transformed.
            points (numpy.ndarray): Point annotations.

        Returns:
            tuple: (transformed_img, transformed_points)
        """
        for t in self.transforms:
            img, points = t(img, points)
        return img, points