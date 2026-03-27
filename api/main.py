"""
main.py
FastAPI backend for the Fruit Freshness Classifier.
Endpoints:
  GET  /           - Health check
  GET  /status     - Model uptime and info
  POST /predict    - Predict freshness from uploaded image
  POST /upload     - Upload new images for retraining
  POST /retrain    - Trigger model retraining
  GET  /classes    - Get all class names
"""

import os
import sys

# Disable Metal GPU to avoid floating point precision issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import shutil
from datetime import datetime
from typing import Optional

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add src/ to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import load_model, load_class_names, retrain_model
from prediction import predict_from_bytes

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fruit Freshness Classifier API",
    description="MLOps API for predicting fruit freshness and triggering retraining.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ───────────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join(os.path.dirname(__file__), '..', 'models', 'fruit_model_final.keras')
UPLOAD_DIR     = os.path.join(os.path.dirname(__file__), '..', 'data', 'retrain_uploads')
START_TIME     = datetime.utcnow()
model          = None
class_names    = None
retrain_status = {"is_retraining": False, "last_retrain": None, "last_metrics": None}


def get_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH)
    return model


def get_class_names():
    global class_names
    if class_names is None:
        class_names = load_class_names(
            os.path.join(os.path.dirname(__file__), '..', 'models', 'class_names.json')
        )
    return class_names


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Fruit Freshness Classifier API is running!",
        "docs": "/docs",
        "status": "/status",
    }


@app.get("/status")
def get_status():
    uptime_seconds = (datetime.utcnow() - START_TIME).total_seconds()
    hours, remainder = divmod(int(uptime_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    return {
        "status": "online",
        "uptime": f"{hours}h {minutes}m {seconds}s",
        "uptime_seconds": uptime_seconds,
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "started_at": START_TIME.isoformat(),
        "is_retraining": retrain_status["is_retraining"],
        "last_retrain": retrain_status["last_retrain"],
        "last_metrics": retrain_status["last_metrics"],
    }


@app.get("/classes")
def get_classes():
    return {"classes": get_class_names()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPG or PNG image."
        )

    try:
        image_bytes = await file.read()
        m = get_model()
        result = predict_from_bytes(m, image_bytes)
        result["filename"] = file.filename
        result["timestamp"] = datetime.utcnow().isoformat()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/upload")
async def upload_images(
    files: list[UploadFile] = File(...),
    class_name: str = Form(...)
):
    valid_classes = get_class_names()
    if class_name not in valid_classes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid class '{class_name}'. Valid classes: {valid_classes}"
        )

    class_upload_dir = os.path.join(UPLOAD_DIR, class_name)
    os.makedirs(class_upload_dir, exist_ok=True)

    saved_files = []
    for file in files:
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            continue
        file_path = os.path.join(class_upload_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        saved_files.append(file.filename)

    return {
        "message": f"Successfully uploaded {len(saved_files)} images for class '{class_name}'",
        "class": class_name,
        "saved_files": saved_files,
        "upload_dir": class_upload_dir,
    }


@app.post("/retrain")
def trigger_retrain(epochs: int = 5):
    if retrain_status["is_retraining"]:
        raise HTTPException(status_code=409, detail="Retraining already in progress.")

    if not os.path.exists(UPLOAD_DIR) or not os.listdir(UPLOAD_DIR):
        raise HTTPException(
            status_code=400,
            detail="No uploaded data found. Please upload images first via /upload."
        )

    try:
        retrain_status["is_retraining"] = True
        m = get_model()
        metrics = retrain_model(m, UPLOAD_DIR, epochs=epochs, save_path=MODEL_PATH)

        retrain_status["is_retraining"] = False
        retrain_status["last_retrain"] = datetime.utcnow().isoformat()
        retrain_status["last_metrics"] = metrics

        global model
        model = load_model(MODEL_PATH)

        return {
            "message": "Retraining completed successfully!",
            "metrics": metrics,
            "timestamp": retrain_status["last_retrain"],
        }
    except Exception as e:
        retrain_status["is_retraining"] = False
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)