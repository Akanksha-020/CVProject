from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import joblib
import numpy as np

from src.config import MODEL_PATH
from src.features import detect_largest_face, extract_features_from_face


def _load_artifact(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Train first with: python main.py train")
    return joblib.load(model_path)


def predict_image(image_path: str, model_path: Path = MODEL_PATH) -> Tuple[str, float]:
    artifact = _load_artifact(model_path)
    pipeline = artifact["pipeline"]
    label_encoder = artifact["label_encoder"]

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_roi = detect_largest_face(gray, face_cascade)
    if face_roi is None:
        face_roi = gray

    features = extract_features_from_face(face_roi).reshape(1, -1)
    pred_idx = int(pipeline.predict(features)[0])
    label = str(label_encoder.inverse_transform([pred_idx])[0])

    confidence = 1.0
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(features)[0]
        confidence = float(np.max(proba))

    return label, confidence
