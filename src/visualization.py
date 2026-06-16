"""Cluster visualization via collated image grids.

Each row represents one cluster, with M sampled images per row.
Supports both representative sampling (closest to centroid) and
random sampling for cluster validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image


def visualize_clusters(
    cluster_images: Dict[int, List[str]],
    image_dir: str | Path,
    grid_size: int = 7,
    thumbnail_size: Tuple[int, int] = (100, 100),
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Cluster Visualization",
    save_path: Optional[str | Path] = None,
) -> None:
    """Display a collated image grid of clustered images.

    Each row corresponds to one cluster, showing ``grid_size`` images.
    Rows are labeled with the cluster index.

    Args:
        cluster_images: Dict mapping cluster_id to list of image filenames.
        image_dir: Directory containing the source images.
        grid_size: Number of images per row (columns).
        thumbnail_size: (width, height) to resize each thumbnail.
        figsize: Matplotlib figure size.
        title: Plot title.
        save_path: Optional path to save the figure. If None, displays
            the plot interactively.
    """
    image_dir = Path(image_dir)
    num_clusters = len(cluster_images)

    fig, axs = plt.subplots(num_clusters, grid_size, figsize=figsize)

    # Handle single-cluster edge case
    if num_clusters == 1:
        axs = [axs]

    for row_idx, (cluster_id, filenames) in enumerate(sorted(cluster_images.items())):
        for col_idx in range(grid_size):
            ax = axs[row_idx][col_idx] if num_clusters > 1 else axs[col_idx]

            if col_idx < len(filenames):
                try:
                    img = Image.open(image_dir / filenames[col_idx]).convert("RGB")
                    img = img.resize(thumbnail_size)
                    ax.imshow(mpimg.pil_to_array(img))
                except (IOError, OSError):
                    ax.set_facecolor("lightgray")
            else:
                ax.set_facecolor("lightgray")

            ax.axis("off")

        # Label the row
        row_ax = axs[row_idx][0] if num_clusters > 1 else axs[0]
        row_ax.set_ylabel(f"C{cluster_id}", fontsize=10, rotation=0, labelpad=20)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
