"""
Utility functions for data loading, evaluation metrics, and miscellaneous operations.
"""

import os
import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import confusion_matrix
from typing import Tuple, List, Dict, Optional


def load_dataset(dataset_name: str, train_ratio: float):
    """
    Load hyperspectral dataset and split into train/test sets.

    Args:
        dataset_name: One of 'IN', 'PU', 'SV', 'HC'
        train_ratio: Proportion of training samples per class (e.g., 0.1 for 10%)
    Returns:
        data_hsi: Normalized HSI cube (H, W, C)
        gt_hsi: Ground truth map (H, W)
        num_classes: Number of classes
        train_pos: (N_train, 2) array of training pixel coordinates
        test_pos: (N_test, 2) array of test pixel coordinates
        train_labels: List of labels corresponding to train_pos
        test_labels: List of labels corresponding to test_pos
    """
    dataset_paths = {
        'IN': ('datasets/Indian_pines_corrected.mat', 'indian_pines_corrected',
               'datasets/Indian_pines_gt.mat', 'indian_pines_gt'),
        'PU': ('datasets/PaviaU.mat', 'paviaU',
               'datasets/PaviaU_gt.mat', 'paviaU_gt'),
        'SV': ('datasets/Salinas_corrected.mat', 'salinas_corrected',
               'datasets/Salinas_gt.mat', 'salinas_gt'),
        'HC': ('datasets/WHU_Hi_HanChuan.mat', 'WHU_Hi_HanChuan',
               'datasets/WHU_Hi_HanChuan_gt.mat', 'WHU_Hi_HanChuan_gt'),
    }
    if dataset_name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data_path, data_key, gt_path, gt_key = dataset_paths[dataset_name]
    data = sio.loadmat(data_path)[data_key]
    gt = sio.loadmat(gt_path)[gt_key]

    # Normalize each band to [0,1]
    data_norm = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[2]):
        band = data[:, :, i]
        min_val, max_val = band.min(), band.max()
        if max_val > min_val:
            data_norm[:, :, i] = (band - min_val) / (max_val - min_val)
        else:
            data_norm[:, :, i] = band

    num_classes = int(gt.max())
    train_pos, test_pos = [], []
    train_labels, test_labels = [], []

    for cls in range(1, num_classes + 1):
        coords = np.argwhere(gt == cls)
        np.random.shuffle(coords)
        n_train = max(1, int(len(coords) * train_ratio))
        train_pos.extend(coords[:n_train])
        test_pos.extend(coords[n_train:])
        train_labels.extend([cls - 1] * n_train)
        test_labels.extend([cls - 1] * (len(coords) - n_train))

    train_pos = np.array(train_pos, dtype=int)
    test_pos = np.array(test_pos, dtype=int)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return data_norm, gt, num_classes, train_pos, test_pos, train_labels, test_labels


def mirror_hsi(data: np.ndarray, patch_size: int):
    """
    Create mirrored version of HSI to handle boundary patches.

    Args:
        data: Original HSI cube (H, W, C)
        patch_size: Size of the patch (must be odd)
    Returns:
        mirrored: Mirrored cube of size (H + 2*pad, W + 2*pad, C) where pad = patch_size // 2 + 4
        pad: padding amount
    """
    pad = patch_size // 2 + 4
    h, w, c = data.shape
    mirrored = np.zeros((h + 2 * pad, w + 2 * pad, c), dtype=data.dtype)
    mirrored[pad:pad + h, pad:pad + w, :] = data
    # Mirror left/right
    for i in range(pad):
        mirrored[pad:pad + h, i, :] = data[:, pad - i - 1, :]
        mirrored[pad:pad + h, w + pad + i, :] = data[:, w - 1 - i, :]
    # Mirror top/bottom
    for i in range(pad):
        mirrored[i, :, :] = mirrored[2 * pad - i - 1, :, :]
        mirrored[h + pad + i, :, :] = mirrored[h + pad - 1 - i, :, :]
    return mirrored, pad


def metrics_from_confusion(confusion: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    """
    Calculate OA, AA, Kappa from confusion matrix.
    """
    tp = np.diag(confusion)
    total = confusion.sum()
    oa = tp.sum() / total
    # Per-class accuracy
    pa = tp / confusion.sum(axis=1)
    aa = pa.mean()
    # Kappa
    pe = (confusion.sum(axis=0) * confusion.sum(axis=1)).sum() / (total ** 2)
    kappa = (oa - pe) / (1 - pe + 1e-10)
    return oa, aa, kappa, pa


def save_classification_map(pred_map: np.ndarray, save_path: str, dpi: int = 300):
    """Save classification map as image."""
    import matplotlib.pyplot as plt
    from matplotlib import colors

    num_classes = int(pred_map.max())
    # Define color map (customize as needed)
    colors_list = plt.cm.tab20(np.linspace(0, 1, num_classes + 1))
    cmap = colors.ListedColormap(colors_list[1:])  # skip background

    plt.figure(figsize=(10, 10), dpi=dpi)
    plt.imshow(pred_map, cmap=cmap, interpolation='nearest')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def load_correlation_cache(cache_path: str) -> dict:
    """Load precomputed correlation cache (dict mapping original (x,y) -> list of mirror coordinates)."""
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    print(f"Loaded correlation cache from {cache_path}, entries: {len(cache)}")
    return cache


def save_correlation_cache(cache: dict, cache_path: str):
    """Save correlation cache to file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Saved correlation cache to {cache_path}")