"""
app.py
Streamlit UI for the Fruit Freshness Classifier.
Pages:
  - Dashboard   : Model uptime and status
  - Predict     : Upload image and get prediction
  - Visualize   : Dataset insights and charts
  - Retrain     : Upload new data and trigger retraining
"""

import streamlit as st
import requests
import json
import os
from PIL import Image
import io
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Fruit Freshness Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("Fruit Freshness Classifier")
st.sidebar.markdown("MLOps Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Predict", "Visualize", "Retrain"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**API:** `{API_URL}`")

# ── Helper ─────────────────────────────────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_URL}/status", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("Model Dashboard")
    st.markdown("Real-time monitoring of the Fruit Freshness Classifier.")
    st.markdown("---")

    status = check_api()

    if status:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Status", "Online")
        col2.metric("Uptime", status.get("uptime", "N/A"))
        col3.metric("Model Loaded", "Yes" if status.get("model_loaded") else "No")
        col4.metric("Retraining", "Active" if status.get("is_retraining") else "Idle")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model Info")
            st.json({
                "model_path": status.get("model_path"),
                "started_at": status.get("started_at"),
                "last_retrain": status.get("last_retrain") or "Never",
            })

        with col2:
            st.subheader("Last Retrain Metrics")
            metrics = status.get("last_metrics")
            if metrics:
                st.json(metrics)
            else:
                st.info("No retraining has been performed yet.")

        st.markdown("---")
        st.subheader("Uptime Gauge")
        uptime_hours = status.get("uptime_seconds", 0) / 3600
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(uptime_hours, 2),
            title={"text": "Uptime (hours)"},
            gauge={
                "axis": {"range": [0, 24]},
                "bar": {"color": "#2ecc71"},
                "steps": [
                    {"range": [0, 8],  "color": "#eafaf1"},
                    {"range": [8, 16], "color": "#a9dfbf"},
                    {"range": [16, 24],"color": "#27ae60"},
                ],
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Cannot connect to the API. Make sure the FastAPI server is running.")
        st.code("uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.title("Predict Fruit Freshness")
    st.markdown("Upload an image of a fruit to classify it as fresh or rotten.")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Choose a fruit image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of an apple, banana, or orange."
    )

    if uploaded_file:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Prediction Result")

            with st.spinner("Analyzing image..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_URL}/predict", files=files, timeout=30)

                    if response.status_code == 200:
                        result = response.json()

                        status_label = "FRESH" if result["status"] == "fresh" else "ROTTEN"
                        st.markdown(f"### {result['predicted_class'].upper()}")
                        st.metric("Confidence", f"{result['confidence']:.2f}%")
                        st.metric("Status", status_label)

                        st.markdown("---")

                        st.subheader("Class Probabilities")
                        probs = result["all_probabilities"]
                        df = pd.DataFrame({
                            "Class": list(probs.keys()),
                            "Probability (%)": list(probs.values())
                        }).sort_values("Probability (%)", ascending=True)

                        colors = ["#2ecc71" if "fresh" in c else "#e74c3c" for c in df["Class"]]
                        fig = px.bar(
                            df, x="Probability (%)", y="Class",
                            orientation="h", color="Class",
                            color_discrete_sequence=colors,
                        )
                        fig.update_layout(showlegend=False, height=300)
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure the server is running.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: VISUALIZE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Visualize":
    st.title("Dataset Visualizations")
    st.markdown("Insights from the Fruit Freshness dataset.")
    st.markdown("---")

    st.subheader("Feature 1: Class Distribution")
    train_data = {
        "freshapples": 1693, "freshbanana": 1581, "freshoranges": 1466,
        "rottenapples": 2342, "rottenbanana": 2224, "rottenoranges": 1595,
    }
    test_data = {
        "freshapples": 395, "freshbanana": 381, "freshoranges": 388,
        "rottenapples": 601, "rottenbanana": 530, "rottenoranges": 403,
    }

    col1, col2 = st.columns(2)
    with col1:
        colors = ["#2ecc71" if "fresh" in k else "#e74c3c" for k in train_data.keys()]
        fig = go.Figure(go.Bar(
            x=list(train_data.keys()),
            y=list(train_data.values()),
            marker_color=colors,
        ))
        fig.update_layout(title="Training Set", xaxis_tickangle=-30, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        colors_t = ["#2ecc71" if "fresh" in k else "#e74c3c" for k in test_data.keys()]
        fig2 = go.Figure(go.Bar(
            x=list(test_data.keys()),
            y=list(test_data.values()),
            marker_color=colors_t,
        ))
        fig2.update_layout(title="Test Set", xaxis_tickangle=-30, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.info("Green = Fresh | Red = Rotten. The dataset is fairly balanced across all 6 classes.")
    st.markdown("---")

    st.subheader("Feature 2: Fresh vs Rotten Split")
    total_fresh  = sum(v for k, v in train_data.items() if "fresh" in k)
    total_rotten = sum(v for k, v in train_data.items() if "rotten" in k)

    fig3 = go.Figure(go.Pie(
        labels=["Fresh", "Rotten"],
        values=[total_fresh, total_rotten],
        marker_colors=["#2ecc71", "#e74c3c"],
        hole=0.4,
    ))
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)
    st.info("The dataset has slightly more rotten samples, reflecting real-world scenarios where spoilage is more varied.")
    st.markdown("---")

    st.subheader("Feature 3: Model Evaluation Metrics per Class")
    metrics_data = {
        "Class": ["freshapples","freshbanana","freshoranges","rottenapples","rottenbanana","rottenoranges"],
        "Precision": [0.99, 1.00, 1.00, 1.00, 1.00, 1.00],
        "Recall":    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        "F1-Score":  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    }
    df_metrics = pd.DataFrame(metrics_data)
    fig4 = px.bar(
        df_metrics.melt(id_vars="Class", var_name="Metric", value_name="Score"),
        x="Class", y="Score", color="Metric", barmode="group",
        color_discrete_map={"Precision": "#3498db", "Recall": "#2ecc71", "F1-Score": "#9b59b6"},
    )
    fig4.update_layout(height=400, xaxis_tickangle=-30, yaxis_range=[0.95, 1.01])
    st.plotly_chart(fig4, use_container_width=True)
    st.info("The model achieves near-perfect scores across all classes, demonstrating strong generalization.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: RETRAIN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Retrain":
    st.title("Retrain Model")
    st.markdown("Upload new images and trigger model retraining.")
    st.markdown("---")

    st.subheader("Step 1: Upload New Images")

    try:
        r = requests.get(f"{API_URL}/classes", timeout=5)
        class_options = r.json().get("classes", [])
    except Exception:
        class_options = ["freshapples","freshbanana","freshoranges","rottenapples","rottenbanana","rottenoranges"]

    selected_class = st.selectbox("Select fruit class for uploaded images:", class_options)
    uploaded_files = st.file_uploader(
        "Upload images for retraining",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Upload Images", type="primary"):
        with st.spinner(f"Uploading {len(uploaded_files)} images..."):
            try:
                files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
                data  = {"class_name": selected_class}
                response = requests.post(f"{API_URL}/upload", files=files, data=data, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    st.success(result["message"])
                    st.json(result)
                else:
                    st.error(f"Upload failed: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.markdown("---")

    st.subheader("Step 2: Trigger Retraining")
    epochs = st.slider("Number of retraining epochs", min_value=1, max_value=20, value=5)
    st.warning("Retraining will take several minutes. Do not close this page.")

    if st.button("Start Retraining", type="primary"):
        with st.spinner("Retraining in progress... please wait."):
            try:
                response = requests.post(
                    f"{API_URL}/retrain",
                    params={"epochs": epochs},
                    timeout=600,
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success("Retraining completed!")
                    st.json(result)
                else:
                    st.error(f"Retraining failed: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Error: {str(e)}")