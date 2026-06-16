"""Feature extraction via transfer learning.

Extracts low-dimensional feature representations from images using
pre-trained ResNet50 (ImageNet) and VGGFace (face-domain) models
with the classification head removed and Global Average Pooling applied.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def build_resnet50_extractor(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
) -> Model:
    """Build a ResNet50 feature extractor (ImageNet pre-trained).

    Removes the classification head and applies Global Average Pooling
    to produce 2048-dimensional feature vectors.

    Args:
        input_shape: (H, W, C) of the input images.

    Returns:
        Keras Model mapping images to feature vectors.
    """
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)

    return Model(inputs=base_model.input, outputs=x, name="resnet50_extractor")


def build_vggface_extractor(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
) -> Model:
    """Build a VGGFace feature extractor (face-domain pre-trained).

    Uses a ResNet50 backbone trained on face images. Removes the
    classification head and applies Global Average Pooling.

    Args:
        input_shape: (H, W, C) of the input images.

    Returns:
        Keras Model mapping images to feature vectors.

    Note:
        Requires the ``keras_vggface`` package:
        ``pip install keras_vggface``
    """
    from keras_vggface.vggface import VGGFace

    base_model = VGGFace(model="resnet50", include_top=False, input_shape=input_shape)

    x = base_model.layers[-2].output
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)

    return Model(inputs=base_model.input, outputs=x, name="vggface_extractor")


def extract_features(
    images: np.ndarray,
    extractor: Model,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract feature vectors from a batch of images.

    Args:
        images: Normalized image array of shape (N, H, W, 3).
        extractor: Keras feature extraction model.
        batch_size: Prediction batch size.

    Returns:
        Feature array of shape (N, feature_dim).
    """
    return extractor.predict(images, batch_size=batch_size, verbose=1)
