# Gunakan image Python ringan sebagai base
FROM python:3.11-slim

# Set environment variable untuk cache InsightFace
ENV INSIGHTFACE_HOME=/opt/insightface

# Install dependency sistem untuk OpenCV, Pillow, dan InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Tentukan working directory di container
WORKDIR /app

# Salin file requirements.txt dan install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua source code ke dalam container
COPY . .

# Environment variable default
ENV HOST=0.0.0.0
ENV PORT=8080
ENV CTX_ID=-1
ENV MODEL_NAME=buffalo_l
ENV FACE_THRESH=0.33
ENV DET_GATE=0.6
ENV LIVE_GATE=0.5

# Gunakan Gunicorn (lebih stabil daripada flask dev server)
# pro_faceid_api:app berarti file pro_faceid_api.py dan variabel Flask app
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-b", "0.0.0.0:8080", "pro_faceid_api:app"]
