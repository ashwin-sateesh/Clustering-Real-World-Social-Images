# From Pixels to Politics: Clustering Congressional Social Images

An unsupervised image clustering framework for analyzing Instagram images from Democrat and Republican congress members. Combines transfer learning (ResNet50, VGGFace) and self-supervised learning (rotation pretext task with iterative pseudo-labeling) to categorize political social media imagery without ground-truth labels.

![Results](/results/result_resnet50_kmeans_representative.png)

## Architecture Overview

The framework follows a four-stage pipeline:

**Stage 1 — Preprocessing:** Instagram images are resized to uniform dimensions (128×128 for ResNet50, 224×224 for VGGFace) and normalized with channel-wise mean/std.

**Stage 2 — Feature Extraction:** Three methods extract low-dimensional representations from the raw pixel data:
- **Transfer Learning (ResNet50):** Pre-trained on ImageNet with classification head removed, produces 2048-D vectors via Global Average Pooling. Captures general visual features (objects, scenes, colors).
- **Transfer Learning (VGGFace):** ResNet50 backbone pre-trained on face images, produces face-domain features. Better at distinguishing political portraits, group photos, and face-centric content.
- **Self-Supervised (Rotation Pretext):** Images are rotated by 0°/90°/180°/270° and a fresh ResNet50 is trained to predict the rotation angle. The learned features are then iteratively refined: extract features → K-Means → pseudo-labels → retrain → repeat. Final model is applied to original unrotated images.

**Stage 3 — Clustering:** K-Means clustering (K=7) groups the feature vectors into coherent visual categories.

**Stage 4 — Validation and Interpretation:** Clusters are validated via two sampling strategies:
- **Representative sampling** — images closest to each centroid (within-cluster consistency)
- **Random sampling** — random draws from each cluster (diversity check)

Manual inspection reveals themes like tweets/graphics, indoor meetings, outdoor rallies, portraits, group photos, and press events.

## Project Structure

```
clustering-real-world-social-images/
├── configs/
│   ├── __init__.py
│   └── config.py              # PathConfig, PreprocessingConfig, ClusteringConfig, etc.
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Image loading, resizing, normalization
│   ├── feature_extraction.py  # ResNet50 and VGGFace transfer learning extractors
│   ├── self_supervised.py     # Rotation pretext task, iterative pseudo-labeling
│   ├── clustering.py          # K-Means, image-cluster mapping, representative sampling
│   └── visualization.py       # Collated image grid generation
├── scripts/
│   └── run_pipeline.py        # Full pipeline: preprocess → extract → cluster → visualize
├── data/                      # Raw images (not tracked)
├── results/                   # Cluster visualizations (tracked)
├── assets/                    # Architecture diagrams
├── requirements.txt
├── .gitignore
└── README.md
```

## Methods

| Method | Feature Extractor | Feature Dim | Domain |
|---|---|---|---|
| Transfer Learning | ResNet50 (ImageNet) | 2048 | General objects/scenes |
| Transfer Learning | VGGFace (ResNet50 backbone) | 2048 | Face-domain features |
| Self-Supervised | ResNet50 (rotation pretext + iterative refinement) | 1000 | Learned from data |

## Setup

```bash
git clone https://github.com/ashwin-sateesh/social-image-clustering.git
cd social-image-clustering

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Data Preparation

Download the congressional Instagram dataset and extract all images into `data/all/`:

```bash
# The original dataset (top20account.zip) contains D/ and R/ subdirectories
# with per-account archives. Extract and flatten all images:
mkdir -p data/all
# Extract all per-account archives into data/all/
```

## Usage

Run the full pipeline:

```bash
python scripts/run_pipeline.py --images-dir ./data/all --results-dir ./results
```

Run specific methods only:

```bash
# ResNet50 transfer learning only
python scripts/run_pipeline.py --images-dir ./data/all --methods resnet50

# Self-supervised only with 10 clusters
python scripts/run_pipeline.py --images-dir ./data/all --methods self_supervised --n-clusters 10

# Skip VGGFace if keras_vggface is not installed
python scripts/run_pipeline.py --images-dir ./data/all --skip-vggface
```

## Cluster Findings

Across all three methods, the most consistent cluster themes identified were:
- **Tweets/graphics** — screenshots of social media posts and infographics
- **Indoor meetings** — office settings, committee hearings
- **Outdoor gatherings** — rallies, press events, public appearances
- **Portraits/headshots** — individual member photos
- **Group photos** — multi-person staged photos
- **Press events** — podium speeches, microphone setups

The ResNet50 transfer learning method produced the most interpretable clusters, with tweet images and group photos forming clearly distinct categories. VGGFace excelled at separating face-centric content (portraits vs group photos). The self-supervised method captured broader visual patterns but with less semantic coherence given the limited training iterations.

## Key Design Decisions

- **VGGFace for political imagery**: Congressional Instagram contains many face-centric images (portraits, meetings). A face-domain pre-trained model captures features that ImageNet-trained models miss, such as distinguishing individual portraits from group settings.
- **Rotation as pretext task**: Predicting rotation forces the network to learn orientation-invariant features without any labels. This is particularly useful since the dataset has no ground-truth categories.
- **Iterative pseudo-labeling**: A single pass of K-Means on self-supervised features produces noisy clusters. Iterating (cluster → retrain → re-extract → re-cluster) refines both the features and the pseudo-labels, progressively improving cluster coherence.
- **Dual validation (representative + random)**: Representative sampling shows within-cluster consistency (are the best examples coherent?), while random sampling reveals cluster diversity and potential noise.

## References

- Zhang & Peng, "Image Clustering: An Unsupervised Approach to Categorize Visual Data in Social Science Research" ([Journal of Communication](https://journalqd.org/article/view/2574))
- He et al. (2015), "Deep Residual Learning for Image Recognition" ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385))
- Gidaris et al. (2018), "Unsupervised Representation Learning by Predicting Image Rotations" ([arXiv:1803.07728](https://arxiv.org/abs/1803.07728))

## License

This project was developed as part of coursework at Northeastern University.
