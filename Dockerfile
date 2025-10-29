# =========================
# Face ID API - Cloud Run
# Base: Python 3.10 (lebih kompatibel utk wheel ML)
# =========================
FROM python:3.10-slim

# Env dasar
ENV INSIGHTFACE_HOME=/opt/insightface \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Paket OS yang dibutuhkan:
# - g++, build-essential : kompilasi ekstensi C/C++ saat install insightface
# - libgl1, libglib2.0-0 : runtime OpenCV
# - libgomp1            : OpenMP runtime (onnxruntime butuh ini)
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ build-essential libgl1 libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Folder kerja
WORKDIR /app

# Install deps Python lebih dulu (pakai wheel jika ada)
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Salin source code
COPY . .

# Default ENV (bisa dioverride di Cloud Run)
ENV HOST=0.0.0.0 \
    PORT=8080 \
    CTX_ID=-1 \
    MODEL_NAME=buffalo_l \
    FACE_THRESH=0.33 \
    DET_GATE=0.6 \
    LIVE_GATE=0.5

# (opsional saja)
EXPOSE 8080

# Jalankan via Gunicorn (production)
# pro_faceid_api:app = file pro_faceid_api.py mengekspor 'app'
CMD ["gunicorn","-w","2","-k","gthread","-b","0.0.0.0:8080","pro_faceid_api:app"]
