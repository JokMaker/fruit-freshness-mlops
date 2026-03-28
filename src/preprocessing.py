"""
preprocessing.py
Handles all image loading, augmentation, and data generator creation
for the Fruit Freshness Classifier.
"""

import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants 
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42

CLASS_NAMES = [
    "freshapples", "freshbanana", "freshoranges",
    "rottenapples", "rottenbanana", "rottenoranges"
]


def get_train_generator(train_dir: str, validation_split: float = 0.2):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=validation_split,
    )

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=SEED,
        shuffle=True,
    )

    val_gen = datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        seed=SEED,
        shuffle=False,
    )

    return train_gen, val_gen


def get_test_generator(test_dir: str):
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_gen = datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return test_gen


def get_retrain_generators(new_data_dir: str, validation_split: float = 0.2):
    """
    Build retraining generators from uploaded images.
    Always uses all 6 CLASS_NAMES so output shape matches the model.
    Creates empty dirs for missing classes so Keras doesn't error.
    """
    # Ensure all 6 class folders exist (empty ones are fine)
    for c in CLASS_NAMES:
        os.makedirs(os.path.join(new_data_dir, c), exist_ok=True)

    # Check at least one class has images
    present_classes = [
        c for c in CLASS_NAMES
        if len(os.listdir(os.path.join(new_data_dir, c))) > 0
    ]
    if not present_classes:
        raise ValueError(
            "No images found in any class folder. "
            "Please upload images first."
        )

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        horizontal_flip=True,
        validation_split=validation_split,
    )

    retrain_gen = datagen.flow_from_directory(
        new_data_dir,
        target_size=IMG_SIZE,
        batch_size=16,
        class_mode="categorical",
        classes=CLASS_NAMES,   # always all 6 — matches model output
        subset="training",
        seed=SEED,
    )

    retrain_val_gen = datagen.flow_from_directory(
        new_data_dir,
        target_size=IMG_SIZE,
        batch_size=16,
        class_mode="categorical",
        classes=CLASS_NAMES,   # always all 6 — matches model output
        subset="validation",
        seed=SEED,
    )

    return retrain_gen, retrain_val_gen


def preprocess_single_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path)
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    else:
        img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    import io
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    else:
        img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array