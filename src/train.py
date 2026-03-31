from __future__ import annotations

from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import MODEL_DIR, MODEL_PATH
from src.data_loader import load_dataset


def _build_model(model_type: str):
    if model_type == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if model_type == "ann":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=42,
            early_stopping=True,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def train_and_evaluate(model_type: str = "knn", test_size: float = 0.2, random_state: int = 42) -> Tuple[float, Dict]:
    x, y = load_dataset()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    clf = _build_model(model_type)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    print(f"Model: {model_type.upper()}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "model_type": model_type,
        "feature_dim": int(x.shape[1]),
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")

    return accuracy, report
