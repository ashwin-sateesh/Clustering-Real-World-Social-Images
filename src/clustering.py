"""K-Means clustering and cluster analysis utilities.

Provides functions for clustering feature vectors, mapping images
to clusters, and finding representative (closest-to-centroid) samples.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def cluster_features(
    features: np.ndarray,
    n_clusters: int = 7,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, KMeans]:
    """Run K-Means clustering on feature vectors.

    Args:
        features: Feature array of shape (N, feature_dim).
        n_clusters: Number of clusters.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (cluster_labels, cluster_centers, kmeans_model).
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    kmeans.fit(features)
    return kmeans.labels_, kmeans.cluster_centers_, kmeans


def map_images_to_clusters(
    labels: np.ndarray,
    index_map: Dict[int, str],
) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
    """Create image-to-cluster and cluster-to-images mappings.

    Args:
        labels: Cluster label for each image (shape N,).
        index_map: Dict mapping integer index to filename.

    Returns:
        Tuple of (image_to_cluster, cluster_to_images):
            - image_to_cluster: {filename: cluster_id}
            - cluster_to_images: {cluster_id: [filenames]}
    """
    img_to_cluster: Dict[str, int] = {}
    cluster_to_imgs: Dict[int, List[str]] = defaultdict(list)

    for idx, label in enumerate(labels):
        filename = index_map[idx]
        img_to_cluster[filename] = int(label)
        cluster_to_imgs[int(label)].append(filename)

    return img_to_cluster, dict(cluster_to_imgs)


def find_representative_samples(
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    index_map: Dict[int, str],
    top_k: int = 7,
) -> Dict[int, List[str]]:
    """Find the top-K images closest to each cluster centroid.

    This implements representative sampling — selecting the most
    typical images from each cluster for visualization and validation.

    Args:
        features: Feature array of shape (N, feature_dim).
        labels: Cluster labels of shape (N,).
        centers: Cluster centroids of shape (K, feature_dim).
        index_map: Dict mapping integer index to filename.
        top_k: Number of closest images per cluster.

    Returns:
        Dict mapping cluster_id to list of closest image filenames.
    """
    distances = cdist(features, centers, "euclidean")
    closest: Dict[int, List[str]] = defaultdict(list)

    for cluster_idx in range(len(centers)):
        cluster_distances = distances[:, cluster_idx]
        closest_indices = np.argsort(cluster_distances)[:top_k]

        for idx in closest_indices:
            filename = index_map[idx]
            cluster_id = int(labels[idx])
            closest[cluster_id].append(filename)

    return dict(closest)


def random_sample_from_clusters(
    cluster_to_images: Dict[int, List[str]],
    sample_size: int = 7,
    seed: int = 42,
) -> Dict[int, List[str]]:
    """Randomly sample images from each cluster.

    Args:
        cluster_to_images: {cluster_id: [filenames]}.
        sample_size: Number of images to sample per cluster.
        seed: Random seed.

    Returns:
        Dict mapping cluster_id to sampled image filenames.
    """
    random.seed(seed)
    sampled: Dict[int, List[str]] = {}

    for cluster_id, images in cluster_to_images.items():
        k = min(sample_size, len(images))
        sampled[cluster_id] = random.sample(images, k)

    return sampled
