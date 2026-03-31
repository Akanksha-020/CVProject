from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np

from src.config import MODEL_PATH
from src.features import extract_features_from_face


def run_realtime(camera_index: int = 0, model_path: Path = MODEL_PATH) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Train first with: python main.py train")

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    label_encoder = artifact["label_encoder"]

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}.")

    print("Starting webcam... press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y : y + h, x : x + w]
            features = extract_features_from_face(face).reshape(1, -1)

            pred_idx = int(pipeline.predict(features)[0])
            label = str(label_encoder.inverse_transform([pred_idx])[0])

            confidence = 1.0
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(features)[0]
                confidence = float(np.max(proba))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 205, 50), 2)
            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (50, 205, 50),
                2,
            )

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
