# pro_faceid_api.py
import os
import io
import base64
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

# Vision / Face
import cv2
import insightface
from insightface.app import FaceAnalysis

# =========================
# Config (via ENV var)
# =========================
HOST         = os.getenv("HOST", "0.0.0.0")
PORT         = int(os.getenv("PORT", "8000"))
CTX_ID       = int(os.getenv("CTX_ID", "-1"))      # -1=CPU, 0/1.. = GPU index
DET_W        = int(os.getenv("DET_W", "640"))
DET_H        = int(os.getenv("DET_H", "640"))
DET_SIZE     = (DET_W, DET_H)
MODEL_NAME   = os.getenv("MODEL_NAME", "buffalo_l")  # retinaface + arcface(r100)
THRESH       = float(os.getenv("FACE_THRESH", "0.33"))  # distance = 1 - cos_sim
DET_GATE     = float(os.getenv("DET_GATE", "0.60"))     # min det_score
LIVE_GATE    = float(os.getenv("LIVE_GATE", "0.50"))    # min liveness score

# =========================
# App & Model Init
# =========================
app = Flask(__name__)
app_insight = FaceAnalysis(name=MODEL_NAME)

# Prefer GPU if CTX_ID>=0; else CPU
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"] if CTX_ID >= 0 else ["CPUExecutionProvider"]

def prepare_model():
    """
    InsightFace >= 0.7.3 mendukung argumen 'providers'.
    Versi lama: tanpa 'providers'. Kita fallback otomatis.
    """
    try:
        app_insight.prepare(ctx_id=CTX_ID, det_size=DET_SIZE, providers=PROVIDERS)
    except TypeError:
        app_insight.prepare(ctx_id=CTX_ID, det_size=DET_SIZE)

prepare_model()

# =========================
# Utilities
# =========================
def b64_to_rgb(b64: str) -> np.ndarray:
    """Decode base64 (dengan/ tanpa 'data:image/...;base64,') menjadi RGB np.ndarray."""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return np.array(img)

def cosine_distance(a, b):
    """Jarak = 1 - cosine_similarity; paksa input jadi 1D dan validasi ukuran."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 1.0  # jarak besar kalau tidak valid
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(1.0 - float(np.dot(a, b)))


def extract_face_and_feature(rgb: np.ndarray) -> Tuple[Optional[object], Optional[np.ndarray], Optional[float]]:
    """
    Deteksi wajah terbaik (det_score tertinggi), kembalikan:
    (face_obj, embedding(512, float32, L2-normalized), det_score)
    """
    faces = app_insight.get(rgb)
    if not faces:
        return None, None, None
    face = max(faces, key=lambda f: f.det_score)
    feat = face.normed_embedding.astype(np.float32)
    return face, feat, float(face.det_score)

def calc_blur_score(rgb: np.ndarray) -> float:
    """Variance of Laplacian; makin tinggi makin tajam."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# def face_box_ratio(face) -> float:
#     """Rasio luas bbox wajah terhadap luas frame."""
#     x1, y1, x2, y2 = map(int, face.bbox.astype(int))
#     box_area = max(0, x2 - x1) * max(0, y2 - y1)
#     # face.image_size: (width, height)
#     frame_area = float(face.image_size[0] * face.image_size[1] + 1e-9)
#     return float(box_area / frame_area)
def face_box_ratio(face, frame_w: int, frame_h: int) -> float:
    """
    Rasio luas bbox wajah terhadap luas frame.
    Tidak lagi memakai face.image_size (bisa None di beberapa versi).
    """
    bbox = getattr(face, "bbox", None)
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = [int(v) for v in bbox]
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    box_area = box_w * box_h
    frame_area = max(1, frame_w * frame_h)  # hindari bagi nol
    return float(box_area / frame_area)

def liveness_score(rgb: np.ndarray, face) -> float:
    """
    Passive liveness heuristic: ketajaman + ukuran wajah.
    Pakai ukuran frame dari rgb.shape, bukan face.image_size.
    """
    h, w = rgb.shape[:2]
    # Blur score (ketajaman)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())      # ~0..1000+
    # Face ratio
    ratio = face_box_ratio(face, w, h)                       # ~0..0.9

    # Normalisasi kasar
    blur_norm  = min(1.0, blur / 300.0)   # >=300 dianggap tajam
    ratio_norm = min(1.0, ratio / 0.12)   # >=12% frame dianggap cukup besar

    return float(0.6 * blur_norm + 0.4 * ratio_norm)

# =========================
# Endpoints
# =========================
@app.get("/")
def root():
    return jsonify({"ok": True, "service": "face_recognition_service"})
@app.post("/health")
def health():
    return jsonify({
        "ok": True,
        "insightface_version": insightface.__version__,
        "model_name": MODEL_NAME,
        "ctx_id": CTX_ID,
        "det_size": DET_SIZE,
        "distance_threshold": THRESH,
        "det_gate": DET_GATE,
        "live_gate": LIVE_GATE,
        "providers": PROVIDERS,
    })

@app.post("/embed")
def embed():
    """
    Ekstrak embedding 512-d + det_score + liveness_score dari sebuah gambar.
    Body:
      { "image_base64": "<B64>" }
    Return:
      {
        ok, det_score, liveness_score,
        embedding: [512 floats]
      }
    """
    try:
        data = request.get_json(force=True)
        rgb = b64_to_rgb(data["image_base64"])
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Invalid image/base64: {e}"}), 400

    face, feat, det_score = extract_face_and_feature(rgb)
    if feat is None:
        return jsonify({"ok": False, "msg": "No face detected"}), 400

    live = liveness_score(rgb, face)
    return jsonify({
        "ok": True,
        "det_score": det_score,
        "liveness_score": live,
        "embedding": [float(x) for x in feat.tolist()]
    })

@app.post("/compare")
def compare():
    """
    Bandingkan foto (probe) dengan satu/lebih template embedding (registered).
    Body:
      {
        "image_base64": "<B64>",
        "templates": [[512 floats], [512 floats], ...]   # minimal 1
      }
    Return:
      {
        ok, det_score, liveness_score,
        best_distance, match, threshold,
        gates: { det_ok, live_ok }
      }
    """
    try:
        data = request.get_json(force=True)
        rgb = b64_to_rgb(data["image_base64"])
        templates = data.get("templates", [])
        if not templates:
            return jsonify({"ok": False, "msg": "No templates provided"}), 400
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Invalid payload: {e}"}), 400

    face, feat, det_score = extract_face_and_feature(rgb)
    if feat is None:
        return jsonify({"ok": False, "msg": "No face detected"}), 400

    # hitung min distance
    dmin = 1e9
    for t in templates:
        t = np.asarray(t, dtype=np.float32)
        d = cosine_distance(feat, t)
        if d < dmin:
            dmin = d

    # gates
    live = liveness_score(rgb, face)
    det_ok  = det_score >= DET_GATE
    live_ok = live      >= LIVE_GATE
    match   = (dmin < THRESH) and det_ok and live_ok

    return jsonify({
        "ok": True,
        "det_score": det_score,
        "liveness_score": live,
        "best_distance": dmin,
        "match": bool(match),
        "threshold": THRESH,
        "gates": {"det_ok": det_ok, "live_ok": live_ok}
    })

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Jalankan:  python pro_faceid_api.py
    # ENV contoh:
    #   CTX_ID=0 FACE_THRESH=0.33 DET_GATE=0.6 LIVE_GATE=0.5 python pro_faceid_api.py
    app.run(host=HOST, port=PORT)
# pro_faceid_api.py
# pro_faceid_api.py
# import os, io, base64
# from typing import Optional, Tuple
# import numpy as np
# from PIL import Image
# from flask import Flask, request, jsonify

# # Import modul berat boleh tetap di sini (aman), tapi JANGAN prepare model di startup
# import cv2
# import insightface
# from insightface.app import FaceAnalysis

# # =========================
# # Config
# # =========================
# HOST         = os.getenv("HOST", "0.0.0.0")
# PORT         = int(os.getenv("PORT", "8000"))
# CTX_ID       = int(os.getenv("CTX_ID", "-1"))          # -1=CPU, >=0 GPU index
# DET_W        = int(os.getenv("DET_W", "640"))
# DET_H        = int(os.getenv("DET_H", "640"))
# DET_SIZE     = (DET_W, DET_H)
# MODEL_NAME   = os.getenv("MODEL_NAME", "buffalo_l")    # retinaface + arcface
# THRESH       = float(os.getenv("FACE_THRESH", "0.33")) # distance = 1 - cos_sim
# DET_GATE     = float(os.getenv("DET_GATE", "0.60"))
# LIVE_GATE    = float(os.getenv("LIVE_GATE", "0.50"))

# # =========================
# # App & Lazy Model
# # =========================
# app = Flask(__name__)
# app_insight = None
# PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"] if CTX_ID >= 0 else ["CPUExecutionProvider"]
# _MODEL_READY = False

# def prepare_model():
#     global app_insight, _MODEL_READY
#     if _MODEL_READY and app_insight is not None:
#         return
#     # siapkan FaceAnalysis saat pertama kali dibutuhkan
#     app_insight = FaceAnalysis(name=MODEL_NAME)
#     try:
#         app_insight.prepare(ctx_id=CTX_ID, det_size=DET_SIZE, providers=PROVIDERS)
#     except TypeError:
#         app_insight.prepare(ctx_id=CTX_ID, det_size=DET_SIZE)
#     _MODEL_READY = True

# # =========================
# # Utilities
# # =========================
# def b64_to_rgb(b64: str) -> np.ndarray:
#     if "," in b64:
#         b64 = b64.split(",", 1)[1]
#     img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
#     return np.array(img)

# def cosine_distance(a, b) -> float:
#     a = np.asarray(a, dtype=np.float32).reshape(-1)
#     b = np.asarray(b, dtype=np.float32).reshape(-1)
#     if a.size == 0 or b.size == 0 or a.size != b.size:
#         return 1.0
#     a = a / (np.linalg.norm(a) + 1e-9)
#     b = b / (np.linalg.norm(b) + 1e-9)
#     return float(1.0 - float(np.dot(a, b)))

# def extract_face_and_feature(rgb: np.ndarray) -> Tuple[Optional[object], Optional[np.ndarray], Optional[float]]:
#     faces = app_insight.get(rgb)
#     if not faces:
#         return None, None, None
#     face = max(faces, key=lambda f: f.det_score)
#     feat = face.normed_embedding.astype(np.float32)
#     return face, feat, float(face.det_score)

# def face_box_ratio(face, frame_w: int, frame_h: int) -> float:
#     bbox = getattr(face, "bbox", None)
#     if bbox is None:
#         return 0.0
#     x1, y1, x2, y2 = [int(v) for v in bbox]
#     box_w = max(0, x2 - x1)
#     box_h = max(0, y2 - y1)
#     box_area = box_w * box_h
#     frame_area = max(1, frame_w * frame_h)
#     return float(box_area / frame_area)

# def liveness_score(rgb: np.ndarray, face) -> float:
#     h, w = rgb.shape[:2]
#     gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
#     blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
#     ratio = face_box_ratio(face, w, h)
#     blur_norm  = min(1.0, blur / 300.0)
#     ratio_norm = min(1.0, ratio / 0.12)
#     return float(0.6 * blur_norm + 0.4 * ratio_norm)

# # =========================
# # Endpoints
# # =========================
# @app.get("/")
# def root():
#     return jsonify({"ok": True, "service": "face_recognition_service"})

# # jadikan GET & ringan: tidak memanggil prepare_model()
# @app.get("/health")
# def health():
#     return jsonify({
#         "ok": True,
#         "insightface_version": insightface.__version__,
#         "model_name": MODEL_NAME,
#         "ctx_id": CTX_ID,
#         "det_size": DET_SIZE,
#         "distance_threshold": THRESH,
#         "det_gate": DET_GATE,
#         "live_gate": LIVE_GATE,
#         "providers": PROVIDERS,
#         "model_ready": _MODEL_READY
#     })

# @app.post("/embed")
# def embed():
#     try:
#         prepare_model()
#     except Exception as e:
#         return jsonify({"ok": False, "msg": f"Model init failed: {e}"}), 500

#     try:
#         data = request.get_json(force=True)
#         rgb = b64_to_rgb(data["image_base64"])
#     except Exception as e:
#         return jsonify({"ok": False, "msg": f"Invalid image/base64: {e}"}), 400

#     face, feat, det_score = extract_face_and_feature(rgb)
#     if feat is None:
#         return jsonify({"ok": False, "msg": "No face detected"}), 400

#     live = liveness_score(rgb, face)
#     return jsonify({
#         "ok": True,
#         "det_score": det_score,
#         "liveness_score": live,
#         "embedding": [float(x) for x in feat.tolist()]
#     })

# @app.post("/compare")
# def compare():
#     try:
#         prepare_model()
#     except Exception as e:
#         return jsonify({"ok": False, "msg": f"Model init failed: {e}"}), 500

#     try:
#         data = request.get_json(force=True)
#         rgb = b64_to_rgb(data["image_base64"])
#         templates = data.get("templates", [])
#         if not templates:
#             return jsonify({"ok": False, "msg": "No templates provided"}), 400
#     except Exception as e:
#         return jsonify({"ok": False, "msg": f"Invalid payload: {e}"}), 400

#     face, feat, det_score = extract_face_and_feature(rgb)
#     if feat is None:
#         return jsonify({"ok": False, "msg": "No face detected"}), 400

#     dmin = min(cosine_distance(feat, np.asarray(t, dtype=np.float32)) for t in templates)
#     live = liveness_score(rgb, face)
#     det_ok  = det_score >= DET_GATE
#     live_ok = live      >= LIVE_GATE
#     match   = (dmin < THRESH) and det_ok and live_ok

#     return jsonify({
#         "ok": True,
#         "det_score": det_score,
#         "liveness_score": live,
#         "best_distance": dmin,
#         "match": bool(match),
#         "threshold": THRESH,
#         "gates": {"det_ok": det_ok, "live_ok": live_ok}
#     })

# # =========================
# # Main (local only)
# # =========================
# if __name__ == "__main__":
#     app.run(host=HOST, port=PORT)
