"""
prediction.py
Handles single image prediction for the Fruit Freshness Classifier.
"""

import numpy as np
import tensorflow as tf
from preprocessing import preprocess_image_bytes, preprocess_single_image
from model import load_class_names


def predict_from_path(model, image_path: str) -> dict:
    img_array = preprocess_single_image(image_path)
    predictions = model.predict(img_array, verbose=0)
    return _format_prediction(predictions)


def predict_from_bytes(model, image_bytes: bytes) -> dict:
    img_array = preprocess_image_bytes(image_bytes)
    predictions = model.predict(img_array, verbose=0)
    return _format_prediction(predictions)


def _format_prediction(predictions: np.ndarray) -> dict:
    class_names = load_class_names()
    predicted_idx   = int(np.argmax(predictions[0]))
    predicted_class = class_names[predicted_idx]
    confidence      = float(predictions[0][predicted_idx]) * 100
    status = "fresh" if "fresh" in predicted_class.lower() else "rotten"

    all_probabilities = {
        class_names[i]: round(float(predictions[0][i]) * 100, 2)
        for i in range(len(class_names))
    }

    return {
        "predicted_class": predicted_class,
        "status": status,
        "confidence": round(confidence, 2),
        "all_probabilities": all_probabilities,
    }