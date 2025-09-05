# ------------------ app.py ------------------
# Hard-disable GPU/XLA noise BEFORE any TF/Keras import
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # force CPU; avoids cuInit(303)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"           # silence TF INFO/WARN
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import streamlit as st
st.set_page_config(layout="centered")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Prefer tf.keras loader for legacy .h5; fall back to Keras 3 if needed
from tensorflow.keras.models import load_model as tf_load_model
from keras.models import load_model as k_load_model
from keras.applications.inception_v3 import preprocess_input

# ─── Config ────────────────────────────────────────────────────────────────────
HF_REPO_ID = "RishiPTrial/my-model-name"
MODEL_FILE = "drosophila_inceptionv3_classifier.h5"
INPUT_SIZE = 299
STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa"
]

# ─── Load Model (robust to legacy H5) ──────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model from Hugging Face…")
def load_model():
    token = st.secrets.get("HF_TOKEN", None)  # if HF repo is private
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE, token=token)

    # 1) Try tf.keras legacy H5 loader (best chance for old graphs)
    try:
        m = tf_load_model(model_path, compile=False)
        return m
    except Exception as e_tf:
        st.warning(f"tf.keras load failed, trying Keras 3 legacy loader…\n{e_tf}")

    # 2) Fall back to Keras 3 H5 loader
    try:
        m = k_load_model(model_path, compile=False)
        return m
    except Exception as e_k:
        st.error(
            "Model load failed with both loaders.\n\n"
            f"tf.keras error:\n{e_tf}\n\nKeras 3 error:\n{e_k}\n\n"
            "Tip: Load once locally with tf.keras and re-save as .keras or SavedModel."
        )
        st.stop()

model = load_model()

# ─── Image Preprocessing ───────────────────────────────────────────────────────
def preprocess_image(pil: Image.Image) -> np.ndarray:
    pil = pil.resize((INPUT_SIZE, INPUT_SIZE)).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32)
    return preprocess_input(arr)  # InceptionV3 expects [-1, 1]

# ─── Prediction ────────────────────────────────────────────────────────────────
def classify(pil: Image.Image):
    arr = preprocess_image(pil)
    preds = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(preds))
    return STAGE_LABELS[idx], float(preds[idx])

# ─── UI ────────────────────────────────────────────────────────────────────────
st.title("Live Drosophila Detection")
st.caption("Live per-frame predictions. If the camera fails to connect, use the snapshot fallback below.")

# ─── Video Processor (no stability; per-frame inference) ───────────────────────
class SimpleProcessor(VideoProcessorBase):
    def __init__(self):
        try:
            self.font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        except Exception:
            self.font = ImageFont.load_default()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)

        # Inference (per frame)
        try:
            label, conf = classify(pil)
        except Exception:
            label, conf = "error", 0.0

        # Draw overlay
        draw = ImageDraw.Draw(pil)
        text = f"{label} ({conf:.0%})"
        try:
            x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=self.font)
        except Exception:
            w, h = draw.textsize(text, font=self.font)
            x0, y0, x1, y1 = 0, 0, w, h
        pad = 6
        draw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill="black")
        draw.text((0, 0), text, font=self.font, fill="red")

        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

# ─── Start Webcam (host-only ICE to avoid STUN/UDP retries) ───────────────────
rtc_cfg = {"iceServers": []}  # good for localhost/LAN; use TURN for internet
webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=SimpleProcessor,
    async_processing=False,          # avoid event-loop/thread races
    rtc_configuration=rtc_cfg
)

# ─── Snapshot fallback (works if WebRTC is blocked) ────────────────────────────
st.divider()
snap = st.camera_input("No luck with live video? Use the snapshot fallback:")
if snap is not None:
    pil = Image.open(snap)
    label, conf = classify(pil)
    st.success(f"{label} ({conf:.0%})")
    st.image(pil, caption="Snapshot prediction")
# ---------------- End app.py ---------------
