from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.config import CLASSES, RAW_DATA_DIR
from src.features import detect_largest_face, extract_features_from_face


def _iter_images(folder: Path):
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        yield from folder.glob(pattern)


def load_dataset(raw_dir: Path = RAW_DATA_DIR) -> Tuple[np.ndarray, np.ndarray]:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    features: List[np.ndarray] = []
    labels: List[str] = []

    for cls in CLASSES:
        class_dir = raw_dir / cls
        if not class_dir.exists():
            continue

        image_paths = list(_iter_images(class_dir))
        for image_path in tqdm(image_paths, desc=f"Loading {cls}"):
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_roi = detect_largest_face(gray, face_cascade)
            if face_roi is None:
                face_roi = gray

            feat = extract_features_from_face(face_roi)
            features.append(feat)
            labels.append(cls)

    if not features:
        raise ValueError(
            "No valid training samples found. Check dataset path and ensure faces are visible in images."
        )

    return np.vstack(features), np.array(labels)
