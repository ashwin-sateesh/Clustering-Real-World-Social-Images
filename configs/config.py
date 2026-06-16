"""Centralized configuration for the Social Image Clustering project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class PathConfig:
    """File and directory paths."""

    data_dir: Path = Path("./data")
    results_dir: Path = Path("./results")
    raw_images_dir: str = "all"

    @property
    def images_path(self) -> Path:
        return self.data_dir / self.raw_images_dir


@dataclass
class PreprocessingConfig:
    """Image preprocessing parameters."""

    resnet_input_size: Tuple[int, int] = (128, 128)
    vggface_input_size: Tuple[int, int] = (224, 224)
    normalize_mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    normalize_std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])


@dataclass
class ClusteringConfig:
    """K-Means clustering parameters."""

    n_clusters: int = 7
    random_state: int = 0
    n_init: str = "auto"
    sample_size: int = 7


@dataclass
class SelfSupervisedConfig:
    """Self-supervised learning parameters."""

    num_rotations: int = 4
    initial_classes: int = 4
    num_iterations: int = 3
    epochs_initial: int = 5
    epochs_iterative: int = 5
    batch_size: int = 32
    optimizer: str = "adam"
    input_shape: Tuple[int, int, int] = (128, 128, 3)


@dataclass
class VisualizationConfig:
    """Cluster visualization parameters."""

    grid_size: int = 7
    thumbnail_size: Tuple[int, int] = (100, 100)
    figure_size: Tuple[int, int] = (12, 8)
