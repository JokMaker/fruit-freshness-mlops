# Base image
FROM python:3.11-slim

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Install system dependencies + nginx
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy all project files
COPY --chown=user . /app

# Copy nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port 7860 (required by HF Spaces)
EXPOSE 7860

# Start nginx, FastAPI and Streamlit
CMD ["sh", "-c", "nginx && uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0"]