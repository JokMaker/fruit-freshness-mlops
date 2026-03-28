---
title: Fruit Freshness MLOps
emoji: 🍎
colorFrom: green
colorTo: red
sdk: docker
pinned: false
---

# Fruit Freshness Classifier — MLOps Pipeline

A production-ready MLOps pipeline for classifying fruits as fresh or rotten using VGG16 transfer learning, FastAPI, and Streamlit.

## Project Description

This project demonstrates an end-to-end Machine Learning pipeline for fruit freshness classification. The model classifies images of apples, bananas, and oranges into 6 categories: freshapples, freshbanana, freshoranges, rottenapples, rottenbanana, and rottenoranges.

The pipeline includes data preprocessing, model training, a REST API for predictions, a monitoring dashboard, and support for model retraining with newly uploaded data.

## Demo

- **Video Demo:** YouTube Link ← replace with your YouTube link
- **Live URL:** https://huggingface.co/spaces/JokMaker/fruit-freshness-mlops

## Tech Stack

| Layer | Tool |
|---|---|
| Model | TensorFlow / VGG16 Transfer Learning |
| API | FastAPI |
| UI | Streamlit |
| Containerization | Docker + Docker Compose |
| Cloud | Hugging Face Spaces |
| Load Testing | Locust |

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 99.85% |
| Precision | 99.85% |
| Recall | 99.85% |
| F1 Score | 99.85% |

## Per-Class Results

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| freshapples | 1.00 | 1.00 | 1.00 | 395 |
| freshbanana | 1.00 | 1.00 | 1.00 | 381 |
| freshoranges | 0.99 | 1.00 | 1.00 | 388 |
| rottenapples | 1.00 | 1.00 | 1.00 | 601 |
| rottenbanana | 1.00 | 1.00 | 1.00 | 530 |
| rottenoranges | 1.00 | 1.00 | 1.00 | 403 |

## Load Testing Results (Locust)

| Users | Avg Response (ms) | 95th Percentile (ms) | RPS | Failures |
|---|---|---|---|---|
| 50 | 92 | 240 | 23.7 | 0% |
| 100 | 143 | 380 | 15.25 | 0% |
| 200 | 1023 | 2400 | 27.29 | 0% |

## Project Structure
```
fruit-freshness-mlops/
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── download_model.py
├── notebook/
│   └── MLPOS.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
├── api/
│   └── main.py
├── ui/
│   └── app.py
├── data/
│   ├── train/
│   └── test/
├── models/
│   ├── fruit_model_final.keras
│   └── class_names.json
└── locust/
    └── locustfile.py
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/JokMaker/fruit-freshness-mlops.git
cd fruit-freshness-mlops
```

### 2. Download the Dataset

Download from Kaggle: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

Place unzipped contents into `data/` folder.

### 3. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Model

The model is automatically downloaded from Hugging Face Hub on startup.
- Model repo: https://huggingface.co/JokMaker/fruit-freshness-vgg16
- Model download (GitHub Release): https://github.com/JokMaker/fruit-freshness-mlops/releases/tag/v1.0.0

### 5. Run Locally

Terminal 1 — Start API:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 — Start UI:
```bash
streamlit run ui/app.py
```

- API docs: http://localhost:8000/docs
- UI: http://localhost:8501

### 6. Run with Docker
```bash
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | Health check |
| GET | /status | Model uptime and status |
| GET | /classes | List all class names |
| POST | /predict | Predict from uploaded image |
| POST | /upload | Upload new images for retraining |
| POST | /retrain | Trigger model retraining |

## Load Testing with Locust
```bash
locust -f locust/locustfile.py --host=http://localhost:8000
```

Open http://localhost:8089 to access the Locust dashboard.

## UI Pages

- **Dashboard** — Model uptime, status, and last retrain metrics
- **Predict** — Upload a fruit image and get a prediction
- **Visualize** — Dataset insights and model evaluation charts
- **Retrain** — Upload new images and trigger retraining

## Dataset

- **Source:** Kaggle — Fruits Fresh and Rotten for Classification
- **Classes:** 6 (freshapples, freshbanana, freshoranges, rottenapples, rottenbanana, rottenoranges)
- **Total Images:** 13,600+
- **Train/Test Split:** ~80/20