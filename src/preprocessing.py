"""Image preprocessing: loading, resizing, and normalization.

Handles the pipeline from raw Instagram images to normalized tensors
ready for feature extraction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def load_and_resize_images(
    image_dir: str | Path,
    target_size: Tuple[int, int] = (128, 128),
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Load all images from a directory and resize to a uniform dimension.

    Args:
        image_dir: Directory containing image files.
        target_size: (width, height) to resize each image.

    Returns:
        Tuple of (image_array, index_to_filename_mapping):
            - image_array: shape (N, H, W, 3) as uint8
            - index_map: dict mapping integer index to filename
    """
    image_dir = Path(image_dir)
    images = []
    index_map: Dict[int, str] = {}
    idx = 0

    for filename in sorted(os.listdir(image_dir)):
        filepath = image_dir / filename
        try:
            img = Image.open(filepath).convert("RGB")
            img = img.resize(target_size)
            images.append(np.array(img))
            index_map[idx] = filename
            idx += 1
        except (IOError, OSError):
            print(f"Skipped invalid image: {filename}")

    return np.array(images), index_map


def normalize_images(
    images: np.ndarray,
    mean: List[float] = None,
    std: List[float] = None,
) -> np.ndarray:
    """Normalize image pixel values to [-1, 1] range.

    Converts uint8 [0, 255] to float32 [0, 1], then applies
    channel-wise mean/std normalization.

    Args:
        images: Image array of shape (N, H, W, 3).
        mean: Per-channel means (default: [0.5, 0.5, 0.5]).
        std: Per-channel standard deviations (default: [0.5, 0.5, 0.5]).

    Returns:
        Normalized array of shape (N, H, W, 3) as float32.
    """
    if mean is None:
        mean = [0.5, 0.5, 0.5]
    if std is None:
        std = [0.5, 0.5, 0.5]

    images = images.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (images - mean) / std
