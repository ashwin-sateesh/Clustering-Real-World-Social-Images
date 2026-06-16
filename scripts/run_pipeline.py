#!/usr/bin/env python3
"""Run the full image clustering pipeline.

Usage:
    python scripts/run_pipeline.py --images-dir ./data/all --results-dir ./results

Runs all three methods:
    1. Transfer learning with ResNet50 (ImageNet)
    2. Transfer learning with VGGFace
    3. Self-supervised with rotation pretext task

Saves cluster visualizations to the results directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import (
    ClusteringConfig,
    PreprocessingConfig,
    SelfSupervisedConfig,
    VisualizationConfig,
)
from src.clustering import (
    cluster_features,
    find_representative_samples,
    map_images_to_clusters,
    random_sample_from_clusters,
)
from src.feature_extraction import (
    build_resnet50_extractor,
    build_vggface_extractor,
    extract_features,
)
from src.preprocessing import load_and_resize_images, normalize_images
from src.self_supervised import (
    create_rotated_dataset,
    iterative_self_supervised,
    predict_original_images,
    train_rotation_predictor,
)
from src.visualization import visualize_clusters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Social Image Clustering Pipeline")
    parser.add_argument("--images-dir", type=str, required=True, help="Directory with all images")
    parser.add_argument("--results-dir", type=str, default="./results", help="Output directory for visualizations")
    parser.add_argument("--n-clusters", type=int, default=7, help="Number of clusters")
    parser.add_argument(
        "--methods", type=str, nargs="+",
        default=["resnet50", "vggface", "self_supervised"],
        choices=["resnet50", "vggface", "self_supervised"],
        help="Which methods to run",
    )
    parser.add_argument("--skip-vggface", action="store_true", help="Skip VGGFace (requires keras_vggface)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prep_cfg = PreprocessingConfig()
    clus_cfg = ClusteringConfig()
    clus_cfg.n_clusters = args.n_clusters
    ss_cfg = SelfSupervisedConfig()
    vis_cfg = VisualizationConfig()
    results_dir = Path(args.results_dir)

    # ======================================================================
    # 1. Preprocessing
    # ======================================================================
    print("=" * 60)
    print("STEP 1: Preprocessing")
    print("=" * 60)

    print(f"Loading images from: {args.images_dir}")
    images_128, index_map = load_and_resize_images(args.images_dir, prep_cfg.resnet_input_size)
    print(f"  Loaded {len(index_map)} images at {prep_cfg.resnet_input_size}")

    images_128_norm = normalize_images(images_128, prep_cfg.normalize_mean, prep_cfg.normalize_std)

    # ======================================================================
    # 2. Transfer Learning — ResNet50
    # ======================================================================
    if "resnet50" in args.methods:
        print("\n" + "=" * 60)
        print("STEP 2a: Transfer Learning — ResNet50 (ImageNet)")
        print("=" * 60)

        extractor_res = build_resnet50_extractor(input_shape=(128, 128, 3))
        features_res = extract_features(images_128_norm, extractor_res)
        print(f"  Feature shape: {features_res.shape}")

        labels_res, centers_res, _ = cluster_features(features_res, clus_cfg.n_clusters)
        _, cluster_to_imgs_res = map_images_to_clusters(labels_res, index_map)
        closest_res = find_representative_samples(features_res, labels_res, centers_res, index_map, clus_cfg.sample_size)
        random_res = random_sample_from_clusters(cluster_to_imgs_res, clus_cfg.sample_size)

        visualize_clusters(
            closest_res, args.images_dir,
            title="ResNet50 — Representative Sampling",
            save_path=results_dir / "resnet50_representative.png",
        )
        visualize_clusters(
            random_res, args.images_dir,
            title="ResNet50 — Random Sampling",
            save_path=results_dir / "resnet50_random.png",
        )

    # ======================================================================
    # 3. Transfer Learning — VGGFace
    # ======================================================================
    if "vggface" in args.methods and not args.skip_vggface:
        print("\n" + "=" * 60)
        print("STEP 2b: Transfer Learning — VGGFace")
        print("=" * 60)

        images_224, _ = load_and_resize_images(args.images_dir, prep_cfg.vggface_input_size)
        images_224_norm = normalize_images(images_224, prep_cfg.normalize_mean, prep_cfg.normalize_std)

        extractor_vgg = build_vggface_extractor(input_shape=(224, 224, 3))
        features_vgg = extract_features(images_224_norm, extractor_vgg)
        print(f"  Feature shape: {features_vgg.shape}")

        labels_vgg, centers_vgg, _ = cluster_features(features_vgg, clus_cfg.n_clusters)
        _, cluster_to_imgs_vgg = map_images_to_clusters(labels_vgg, index_map)
        closest_vgg = find_representative_samples(features_vgg, labels_vgg, centers_vgg, index_map, clus_cfg.sample_size)
        random_vgg = random_sample_from_clusters(cluster_to_imgs_vgg, clus_cfg.sample_size)

        visualize_clusters(
            closest_vgg, args.images_dir,
            title="VGGFace — Representative Sampling",
            save_path=results_dir / "vggface_representative.png",
        )
        visualize_clusters(
            random_vgg, args.images_dir,
            title="VGGFace — Random Sampling",
            save_path=results_dir / "vggface_random.png",
        )

    # ======================================================================
    # 4. Self-Supervised Learning
    # ======================================================================
    if "self_supervised" in args.methods:
        print("\n" + "=" * 60)
        print("STEP 3: Self-Supervised Learning (Rotation Pretext)")
        print("=" * 60)

        print("Creating rotated dataset...")
        rotated_imgs, rot_labels = create_rotated_dataset(images_128_norm, ss_cfg.num_rotations)
        print(f"  Rotated dataset: {rotated_imgs.shape}")

        print("Training rotation predictor...")
        rot_model, rot_extractor = train_rotation_predictor(
            rotated_imgs, rot_labels,
            input_shape=ss_cfg.input_shape,
            num_classes=ss_cfg.initial_classes,
            epochs=ss_cfg.epochs_initial,
            batch_size=ss_cfg.batch_size,
        )

        initial_features = rot_extractor.predict(rotated_imgs, verbose=0)

        print("Running iterative refinement...")
        final_clf, final_ext, _ = iterative_self_supervised(
            rotated_imgs, initial_features,
            input_shape=ss_cfg.input_shape,
            n_clusters=clus_cfg.n_clusters,
            num_iterations=ss_cfg.num_iterations,
            epochs=ss_cfg.epochs_iterative,
            batch_size=ss_cfg.batch_size,
        )

        print("Predicting on original images...")
        features_ss, labels_ss_direct = predict_original_images(images_128_norm, final_clf, final_ext)

        # Also cluster the self-supervised features with K-Means
        labels_ss, centers_ss, _ = cluster_features(features_ss, clus_cfg.n_clusters)
        _, cluster_to_imgs_ss = map_images_to_clusters(labels_ss, index_map)
        closest_ss = find_representative_samples(features_ss, labels_ss, centers_ss, index_map, clus_cfg.sample_size)
        random_ss = random_sample_from_clusters(cluster_to_imgs_ss, clus_cfg.sample_size)

        visualize_clusters(
            closest_ss, args.images_dir,
            title="Self-Supervised — Representative Sampling",
            save_path=results_dir / "self_supervised_representative.png",
        )
        visualize_clusters(
            random_ss, args.images_dir,
            title="Self-Supervised — Random Sampling",
            save_path=results_dir / "self_supervised_random.png",
        )

    print("\n" + "=" * 60)
    print(f"Pipeline complete! Results saved to {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
