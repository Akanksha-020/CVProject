from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from skimage.feature import hog

from src.config import FACE_SIZE


def detect_largest_face(gray_image: np.ndarray, face_cascade: cv2.CascadeClassifier) -> Optional[np.ndarray]:
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return gray_image[y : y + h, x : x + w]


def extract_features_from_face(face_roi: np.ndarray) -> np.ndarray:
    resized = cv2.resize(face_roi, FACE_SIZE)
    normalized = resized.astype("float32") / 255.0

    hog_features = hog(
        normalized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return hog_features.astype("float32")
