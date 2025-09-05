# app.py
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")  # ensure Keras 3 uses TF backend

import streamlit as st
st.set_page_config(layout="centered")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from huggingface_hub import hf_hub_download

# 👉 Use KERAS 3 (not tf.keras)
from keras.models import load_model as k_load_model
from keras.applications.inception_v3 import preprocess_input

# ─── Config ─────────────────────────────────────────────────────────────────────
HF_REPO_ID = "RishiPTrial/my-model-name"
MODEL_FILE = "drosophila_inceptionv3_classifier.h5"
INPUT_SIZE = 299
STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa"
]

# ─── Load Model (Keras 3) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model from Hugging Face…")
def load_model():
    token = st.secrets.get("HF_TOKEN", None)  # if HF repo is private
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE, token=token)
    model = k_load_model(model_path, compile=False)
    return model

model = load_model()

# ─── Image Preprocessing ────────────────────────────────────────────────────────
def preprocess_image(pil: Image.Image) -> np.ndarray:
    pil = pil.resize((INPUT_SIZE, INPUT_SIZE)).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32)
    return preprocess_input(arr)  # InceptionV3 scaling [-1, 1]

# ─── Prediction ─────────────────────────────────────────────────────────────────
def classify(pil: Image.Image):
    arr = preprocess_image(pil)
    preds = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(preds))
    return STAGE_LABELS[idx], float(preds[idx])

# ─── UI ─────────────────────────────────────────────────────────────────────────
st.title("Live Drosophila Detection")
st.caption("If camera doesn’t start, try Chrome on desktop/macOS. A snapshot fallback is provided below.")

# ─── Video Processor ────────────────────────────────────────────────────────────
class StableProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_label = None
        self.count = 0
        self.stable_label = "Waiting..."
        try:
            self.font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        except Exception:
            self.font = ImageFont.load_default()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)

        # Run classifier (guarded)
        try:
            label, conf = classify(pil)
        except Exception:
            label, conf = "error", 0.0

        # Stability over 3 consecutive frames
        if label == self.last_label:
            self.count += 1
        else:
            self.last_label = label
            self.count = 1
        if self.count >= 3:
            self.stable_label = label

        # Draw overlay
        draw = ImageDraw.Draw(pil)
        text = f"{label} ({conf:.0%})"

        try:
            bbox = draw.textbbox((0, 0), text, font=self.font)
            x0, y0, x1, y1 = bbox
        except Exception:
            w, h = draw.textsize(text, font=self.font)
            x0, y0, x1, y1 = 0, 0, w, h

        padding = 6
        draw.rectangle([x0 - padding, y0 - padding, x1 + padding, y1 + padding], fill="black")
        draw.text((0, 0), text, font=self.font, fill="red")

        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

# ─── Start Webcam (No STUN to avoid UDP/ICE retries) ────────────────────────────
rtc_cfg = {"iceServers": []}  # host candidates only; great for localhost/LAN
webrtc_ctx = webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=StableProcessor,
    async_processing=False,                 # avoid event-loop/thread races
    rtc_configuration=rtc_cfg
)

# ─── Display Stable Result ──────────────────────────────────────────────────────
st.markdown("### 🧠 Stable Prediction (after 3 consistent frames):")
if webrtc_ctx and webrtc_ctx.video_processor:
    st.success(webrtc_ctx.video_processor.stable_label)
else:
    st.info("Waiting for camera permission…")

# ─── Snapshot fallback (works even if WebRTC is blocked) ────────────────────────
st.divider()
snap = st.camera_input("No luck with live video? Use the snapshot fallback:")
if snap is not None:
    pil = Image.open(snap)
    label, conf = classify(pil)
    st.success(f"{label} ({conf:.0%})")
    st.image(pil, caption="Snapshot")
