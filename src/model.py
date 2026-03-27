"""
model.py
Handles model building, loading, and retraining
for the Fruit Freshness Classifier.
"""

import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from preprocessing import get_retrain_generators

# Constants
IMG_SIZE         = (224, 224)
NUM_CLASSES      = 6
MODEL_PATH       = "models/fruit_model_final.keras"
CLASS_NAMES_PATH = "models/class_names.json"


def build_model(num_classes: int = NUM_CLASSES) -> keras.Model:
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def load_model(model_path: str = MODEL_PATH) -> keras.Model:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)


def load_class_names(path: str = CLASS_NAMES_PATH) -> list:
    with open(path, "r") as f:
        return json.load(f)


def retrain_model(
    model: keras.Model,
    new_data_dir: str,
    epochs: int = 5,
    learning_rate: float = 1e-5,
    save_path: str = MODEL_PATH,
) -> dict:
    retrain_gen, retrain_val_gen = get_retrain_generators(new_data_dir)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    history = model.fit(
        retrain_gen,
        epochs=epochs,
        validation_data=retrain_val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(save_path)
    print(f"Retrained model saved to {save_path}")

    return {
        "epochs_run": len(history.history["accuracy"]),
        "final_train_accuracy": round(history.history["accuracy"][-1], 4),
        "final_val_accuracy": round(history.history["val_accuracy"][-1], 4),
        "model_path": save_path,
    }