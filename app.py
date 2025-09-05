# app.py
import streamlit as st
st.set_page_config(page_title="Drosophila Stage Identification", layout="centered")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download

# WebRTC (same simple pattern as your working app)
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Keras 3 runtime
from keras.models import load_model as k_load_model
from keras.applications.inception_v3 import preprocess_input

# ─── App config ─────────────────────────────────────────────────────────────────
HF_REPO_ID = "RishiPTrial/my-model-name"
MODEL_FILE = "drosophila_inceptionv3_classifier.h5"
INPUT_SIZE = 299
STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa"
]

# ─── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model from Hugging Face…")
def load_stage_model():
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE, token=token)
    return k_load_model(path, compile=False)

model = load_stage_model()

# ─── Helpers ────────────────────────────────────────────────────────────────────
def preprocess_image(pil: Image.Image) -> np.ndarray:
    pil = pil.resize((INPUT_SIZE, INPUT_SIZE)).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32)
    return preprocess_input(arr)  # InceptionV3 expects [-1, 1] scaling via preprocess_input

def classify(pil: Image.Image):
    x = preprocess_image(pil)[np.newaxis]
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    return STAGE_LABELS[idx], float(preds[idx]), preds

def annotate(pil: Image.Image, text: str) -> Image.Image:
    out = pil.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    # text size (with fallback)
    try:
        bbox = draw.textbbox((0, 0), text, font=font); w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        w, h = draw.textsize(text, font=font)
    pad = 6
    draw.rectangle([0-pad, 0-pad, w+pad, h+pad], fill="black")
    draw.text((0, 0), text, font=font, fill="red")
    return out

def show_probs(labels, probs):
    rows = sorted([(lab, float(p)) for lab, p in zip(labels, probs)],
                  key=lambda x: x[1], reverse=True)
    for lab, p in rows:
        st.write(f"{lab}: {p:.2%}")

# ─── UI ─────────────────────────────────────────────────────────────────────────
st.title("🪰 Drosophila Stage Identification")
st.write("Upload an image or use the live camera (same simple WebRTC pattern as your other app).")

st.markdown("### 📷 Upload Image")
uploaded = st.file_uploader("Upload a Drosophila image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    label, conf, probs = classify(pil)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Prediction")
        st.write(f"**Stage:** {label}")
        st.write(f"**Confidence:** {conf:.2%}")
        with st.expander("Class probabilities"):
            show_probs(STAGE_LABELS, probs)
    with c2:
        st.subheader("Annotated")
        st.image(annotate(pil, f"{label} ({conf:.0%})"), use_column_width=True)

st.markdown("---")
st.subheader("📸 Live Camera Stage Detection")

# ─── Video processor (no Streamlit calls, no session_state mutations) ───────────
class StageProcessor(VideoProcessorBase):
    def __init__(self):
        self.font = None
        try:
            self.font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        except Exception:
            self.font = ImageFont.load_default()
        self.last_label = None  # optional display continuity

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)

        # predict on this frame
        label, conf, _ = classify(pil)
        self.last_label = label

        # draw overlay
        text = f"{label} ({conf:.0%})"
        out = annotate(pil, text)
        return av.VideoFrame.from_ndarray(np.array(out), format="rgb24")

# ─── Start webcam (identical shape to your working app) ─────────────────────────
webrtc_streamer(
    key="live-stage-detect",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=StageProcessor,
    async_processing=True,  # match your working app
)
