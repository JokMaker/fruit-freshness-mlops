"""
download_model.py
Downloads the model and class_names.json from Hugging Face Hub
if they are not already present locally.
"""

import os
from huggingface_hub import hf_hub_download

REPO_ID    = "JokMaker/fruit-freshness-vgg16"
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
MODEL_FILE = os.path.join(MODEL_DIR, "fruit_model_final.keras")
CLASS_FILE = os.path.join(MODEL_DIR, "class_names.json")


def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_FILE):
        print("Downloading model from Hugging Face Hub...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename="fruit_model_final.keras",
            local_dir=MODEL_DIR,
        )
        print(f"Model saved to {MODEL_FILE}")
    else:
        print("Model already exists, skipping download.")

    if not os.path.exists(CLASS_FILE):
        print("Downloading class_names.json from Hugging Face Hub...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename="class_names.json",
            local_dir=MODEL_DIR,
        )
        print(f"Class names saved to {CLASS_FILE}")
    else:
        print("class_names.json already exists, skipping download.")


if __name__ == "__main__":
    download_model()
