"""
SCSP (Spectral-Correlation and Spatial-Pooling) and GPCF (Group-Pooling and Cross-Feature Fusion)
modules. These are non-trainable preprocessing steps that transform the input HSI patch into
compact 1D sequences for the Transformer encoder.
"""

import numpy as np
from itertools import combinations
from scipy.stats import pearsonr
from typing import List, Tuple, Dict, Optional
from utils import mirror_hsi, load_correlation_cache, save_correlation_cache


def pearson_correlation_matrix(patch: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation between central pixel and all pixels in patch.

    Args:
        patch: 2D array of shape (height, width, bands)
    Returns:
        correlation matrix of shape (height, width)
    """
    h, w, b = patch.shape
    center = patch[h // 2, w // 2, :].reshape(1, -1)
    patch_flat = patch.reshape(-1, b)
    # Compute correlations efficiently
    center_mean = center - center.mean()
    patch_mean = patch_flat - patch_flat.mean(axis=1, keepdims=True)
    numerator = (center_mean * patch_mean).sum(axis=1)
    denominator = np.sqrt((center_mean ** 2).sum() * (patch_mean ** 2).sum(axis=1))
    corr = numerator / (denominator + 1e-10)
    return corr.reshape(h, w)


class SCSP:
    """
    Spectral-Correlation and Spatial-Pooling Module.
    Selects top-k correlated pixels, extracts overlapping patches, and performs average pooling.
    """

    def __init__(self, patch_size: int = 7, hpf_patch_size: int = 9, top_k: int = 13):
        """
        Args:
            patch_size: Size of input patch (default 7)
            hpf_patch_size: Size of sub-patch for pooling (default 9)
            top_k: Number of pixels to select (including center)
        """
        self.patch_size = patch_size
        self.hpf_patch_size = hpf_patch_size
        self.top_k = top_k
        self.half = patch_size // 2
        self.hpf_half = hpf_patch_size // 2

    def _select_pixels(self, mirror_image: np.ndarray, center_mirror: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Select top-k correlated pixels based on spectral correlation.
        Returns list of (x, y) coordinates in mirror_image space.
        """
        x, y = center_mirror
        # Extract the patch of size patch_size x patch_size
        patch = mirror_image[x - self.half:x + self.half + 1,
                             y - self.half:y + self.half + 1, :]
        corr = pearson_correlation_matrix(patch)
        # Get indices of top-k correlations (flattened)
        flat_idx = np.argsort(corr.ravel())[-self.top_k:]
        coords = np.array(np.unravel_index(flat_idx, patch.shape[:2])).T
        # Convert to global mirror coordinates
        global_coords = [(x - self.half + fx, y - self.half + fy) for (fx, fy) in coords]
        return global_coords

    def process(self, mirror_image: np.ndarray, center_mirror: Tuple[int, int],
                cache: Optional[Dict] = None, original_center: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Args:
            mirror_image: Mirrored HSI cube (H+2*pad, W+2*pad, C)
            center_mirror: (x, y) coordinates in mirror_image space
            cache: Optional precomputed correlation cache (dict mapping original (x,y) -> mirror coordinates list)
            original_center: Original (x, y) coordinates in original image space (required if cache is provided)
        Returns:
            Feature vectors for selected pixels: shape (top_k, bands)
        """
        if cache is not None and original_center is not None:
            # Use precomputed coordinates from cache
            focal_mirror = cache.get(original_center)
            if focal_mirror is None:
                raise KeyError(f"Center {original_center} not found in cache")
        else:
            # Compute on the fly
            focal_mirror = self._select_pixels(mirror_image, center_mirror)

        vectors = []
        for (fx, fy) in focal_mirror:
            # Extract sub-patch of size hpf_patch_size x hpf_patch_size
            sub_patch = mirror_image[fx - self.hpf_half:fx + self.hpf_half + 1,
                                     fy - self.hpf_half:fy + self.hpf_half + 1, :]
            # Average pooling across spatial dimensions
            pooled = sub_patch.mean(axis=(0, 1))
            vectors.append(pooled)
        return np.array(vectors)  # shape: (top_k, bands)


class GPCF:
    """
    Group-Pooling and Cross-Feature Fusion Module.
    Performs combinatorial grouping, multi-mode pooling, and cross-feature fusion.
    """

    def __init__(self, group_size: int = 4):
        """
        Args:
            group_size: Number of vectors per group (default 4)
        """
        self.group_size = group_size
        # Precompute combinations for indices 0..11 (since we have 12 non-center vectors)
        self.combinations = list(combinations(range(12), group_size))
        self.num_groups = len(self.combinations)

    def group_and_pool(self, scsp_vectors: np.ndarray) -> np.ndarray:
        """
        Args:
            scsp_vectors: Array of shape (13, bands) where last vector is center pixel.
                          First 12 are selected neighbors.
        Returns:
            Fused 1D sequence of shape (num_groups, 4 * bands)
        """
        center = scsp_vectors[-1]  # center spectral vector
        neighbors = scsp_vectors[:-1]  # 12 neighbor vectors

        fused_sequences = []
        for indices in self.combinations:
            group = neighbors[list(indices)]  # (group_size, bands)
            # Compute min, mean, max pooling
            min_pool = group.min(axis=0)
            mean_pool = group.mean(axis=0)
            max_pool = group.max(axis=0)
            # Cross-feature fusion in fixed order: min, mean, center, max
            # Interleaved: for each band j: min_j, mean_j, center_j, max_j
            fused = np.empty(4 * len(center))
            fused[0::4] = min_pool
            fused[1::4] = mean_pool
            fused[2::4] = center
            fused[3::4] = max_pool
            fused_sequences.append(fused)
        return np.array(fused_sequences)  # (num_groups, 4*bands)

    def process(self, scsp_vectors: np.ndarray) -> np.ndarray:
        """Wrapper for group_and_pool."""
        return self.group_and_pool(scsp_vectors)


def precompute_correlation_cache(data: np.ndarray, gt: np.ndarray,
                                 patch_size: int, hpf_patch_size: int,
                                 top_k: int, cache_path: str,
                                 progress_interval: int = 5000):
    """
    Precompute the top-k correlated pixel coordinates for all non-background pixels.
    Saves a dictionary mapping original (x,y) -> list of mirror coordinates.

    Args:
        data: Original HSI cube (H, W, C)
        gt: Ground truth map (H, W), 0 denotes background
        patch_size: Size for SCSP initial patch
        hpf_patch_size: Size for sub-patch pooling
        top_k: Number of pixels to select (including center)
        cache_path: Path to save the cache (.pkl file)
        progress_interval: Print progress every this many pixels
    """
    # Create mirrored image
    mirror_img, pad = mirror_hsi(data, patch_size)
    h, w, _ = data.shape

    # Get all non-background pixel coordinates
    coords = np.argwhere(gt > 0)
    print(f"Precomputing correlation cache for {len(coords)} pixels...")

    scsp = SCSP(patch_size, hpf_patch_size, top_k)
    cache = {}

    for idx, (x, y) in enumerate(coords):
        # Mirror coordinates
        cx = x + pad
        cy = y + pad
        # Compute focal points (mirror coordinates)
        focal_mirror = scsp._select_pixels(mirror_img, (cx, cy))
        # Store with original coordinates as key
        cache[(int(x), int(y))] = focal_mirror

        if (idx + 1) % progress_interval == 0:
            print(f"  Processed {idx+1}/{len(coords)} pixels")

    # Save cache
    save_correlation_cache(cache, cache_path)
    return cache


def preprocess_patch(mirror_image: np.ndarray, pad: int, center_original: Tuple[int, int],
                     patch_size: int, hpf_patch_size: int, top_k: int, group_size: int,
                     cache: Optional[Dict] = None) -> np.ndarray:
    """
    Complete preprocessing pipeline: SCSP + GPCF, with optional cache.

    Args:
        mirror_image: Mirrored HSI cube (H+2*pad, W+2*pad, bands)
        pad: Padding amount used to generate mirror_image
        center_original: (x, y) coordinates in original image space
        patch_size: Size for SCSP initial patch
        hpf_patch_size: Size for sub-patch pooling
        top_k: Number of pixels to select (including center)
        group_size: Group size for GPCF combination
        cache: Optional precomputed correlation cache (original -> mirror coordinates list)
    Returns:
        Final 1D sequence of shape (C(12, group_size), 4*bands)
    """
    # Convert original coordinates to mirror coordinates
    cx = center_original[0] + pad
    cy = center_original[1] + pad

    scsp = SCSP(patch_size, hpf_patch_size, top_k)
    if cache is not None:
        scsp_vectors = scsp.process(mirror_image, (cx, cy), cache=cache, original_center=center_original)
    else:
        scsp_vectors = scsp.process(mirror_image, (cx, cy))

    gpcf = GPCF(group_size=group_size)
    return gpcf.process(scsp_vectors)