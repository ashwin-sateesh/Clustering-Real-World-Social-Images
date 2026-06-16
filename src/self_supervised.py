"""Self-supervised feature learning via rotation prediction.

Implements a self-supervised pipeline:
    1. Data augmentation — rotate images by 0°, 90°, 180°, 270°
    2. Train ResNet50 to predict rotation angle (pretext task)
    3. Extract features → K-Means clustering → pseudo-labels
    4. Retrain ResNet50 on pseudo-labels (iterative refinement)
    5. Apply final model to original (unrotated) images
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model


# ---------------------------------------------------------------------------
# Data augmentation: rotation
# ---------------------------------------------------------------------------

def create_rotated_dataset(
    images: np.ndarray,
    num_rotations: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a rotation-augmented dataset for the pretext task.

    Each image is rotated by 0°, 90°, 180°, and 270°. The rotation
    count (0, 1, 2, 3) serves as the label.

    Args:
        images: Normalized image array of shape (N, H, W, 3).
        num_rotations: Number of rotation variants per image.

    Returns:
        Tuple of (rotated_images, labels):
            - rotated_images: shape (N * num_rotations, H, W, 3)
            - labels: shape (N * num_rotations,) with values in [0, num_rotations)
    """
    all_rotated = []
    all_labels = []

    for i in range(len(images)):
        for k in range(num_rotations):
            rotated = tf.image.rot90(images[i], k=k).numpy()
            all_rotated.append(rotated)
            all_labels.append(k)

    return np.array(all_rotated), np.array(all_labels)


# ---------------------------------------------------------------------------
# Pretext training: predict rotation
# ---------------------------------------------------------------------------

def train_rotation_predictor(
    rotated_images: np.ndarray,
    labels: np.ndarray,
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 4,
    epochs: int = 5,
    batch_size: int = 32,
) -> Tuple[Model, Model]:
    """Train a ResNet50 to predict image rotation as a pretext task.

    Args:
        rotated_images: Augmented images of shape (N, H, W, 3).
        labels: Rotation labels (0-3).
        input_shape: Input image dimensions.
        num_classes: Number of rotation classes.
        epochs: Training epochs.
        batch_size: Training batch size.

    Returns:
        Tuple of (full_model, feature_extractor):
            - full_model: trained classifier
            - feature_extractor: model outputting penultimate layer features
    """
    model = ResNet50(
        include_top=True, weights=None,
        input_shape=input_shape, classes=num_classes,
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(rotated_images, labels, epochs=epochs, batch_size=batch_size, verbose=1)

    feature_extractor = Model(
        inputs=model.input,
        outputs=model.layers[-2].output,
        name="rotation_feature_extractor",
    )
    return model, feature_extractor


# ---------------------------------------------------------------------------
# Iterative pseudo-label refinement
# ---------------------------------------------------------------------------

def iterative_self_supervised(
    rotated_images: np.ndarray,
    initial_features: np.ndarray,
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    n_clusters: int = 7,
    num_iterations: int = 3,
    epochs: int = 5,
    batch_size: int = 32,
) -> Tuple[Model, Model, np.ndarray]:
    """Run the iterative self-supervised refinement loop.

    Alternates between K-Means clustering of features and retraining
    ResNet50 on the resulting pseudo-labels.

    Args:
        rotated_images: Augmented images of shape (N, H, W, 3).
        initial_features: Feature vectors from the rotation pretext model.
        input_shape: Input image dimensions.
        n_clusters: Number of K-Means clusters.
        num_iterations: Number of cluster-retrain iterations.
        epochs: Training epochs per iteration.
        batch_size: Training batch size.

    Returns:
        Tuple of (final_classifier, final_feature_extractor, final_features).
    """
    features = initial_features
    classifier = None
    extractor = None

    for i in range(num_iterations):
        print(f"\n--- Iteration {i + 1}/{num_iterations} ---")

        # Cluster current features
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        kmeans.fit(features)
        pseudo_labels = kmeans.labels_

        # Retrain on pseudo-labels
        classifier = ResNet50(
            include_top=True, weights=None,
            input_shape=input_shape, classes=n_clusters,
        )
        classifier.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        classifier.fit(
            rotated_images, pseudo_labels,
            epochs=epochs, batch_size=batch_size, verbose=1,
        )

        # Extract updated features
        extractor = Model(
            inputs=classifier.input,
            outputs=classifier.layers[-2].output,
            name=f"ss_extractor_iter{i + 1}",
        )
        features = extractor.predict(rotated_images, verbose=0)

    return classifier, extractor, features


def predict_original_images(
    original_images: np.ndarray,
    classifier: Model,
    feature_extractor: Model,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the self-supervised model to original (unrotated) images.

    Args:
        original_images: Normalized images of shape (N, H, W, 3).
        classifier: Final trained classifier from iterative refinement.
        feature_extractor: Corresponding feature extractor.

    Returns:
        Tuple of (features, predicted_labels).
    """
    features = feature_extractor.predict(original_images, verbose=0)
    probs = classifier.predict(original_images, verbose=0)
    labels = np.argmax(probs, axis=1)
    return features, labels
