# app.py
import streamlit as st
st.set_page_config(layout="centered")

# --- Compatibility shim: some streamlit-webrtc versions call st.experimental_rerun,
# --- which may not exist on newer Streamlit. Alias it to st.rerun if missing.
if not hasattr(st, "experimental_rerun") and hasattr(st, "rerun"):
    try:
        st.experimental_rerun = st.rerun  # type: ignore[attr-defined]
    except Exception:
        pass

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

# Basic STUN so WebRTC can discover peers on Cloud
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# ─── Load Model (Keras 3) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model from Hugging Face…")
def load_model():
    token = st.secrets.get("HF_TOKEN", None)  # if HF repo is private
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE, token=token)
        model = k_load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

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
    return STAGE_LABELS[idx], float(preds[idx]), preds

def draw_label(pil: Image.Image, label: str, conf: float) -> Image.Image:
    pil = pil.copy()
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    text = f"{label} ({conf:.0%})"
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        w, h = draw.textsize(text, font=font)

    padding = 6
    bg_rect = [0 - padding, 0 - padding, w + padding, h + padding]
    draw.rectangle(bg_rect, fill="black")
    draw.text((0, 0), text, font=font, fill="red")
    return pil

# ─── UI ─────────────────────────────────────────────────────────────────────────
st.title("Live Drosophila Detection")
st.subheader("📹 Live Camera Detection with Stable Prediction")

if "stable_prediction" not in st.session_state:
    st.session_state["stable_prediction"] = "Waiting..."

# ─── Video Processor ────────────────────────────────────────────────────────────
class StableProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_label = None
        self.count = 0
        self.stable_label = None
        try:
            self.font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        except Exception:
            self.font = ImageFont.load_default()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)

        label, conf, _ = classify(pil)

        if label == self.last_label:
            self.count += 1
        else:
            self.last_label = label
            self.count = 1

        if self.count >= 3:
            self.stable_label = label
            st.session_state["stable_prediction"] = self.stable_label

        draw = ImageDraw.Draw(pil)
        text = f"{label} ({conf:.0%})"
        try:
            bbox = draw.textbbox((0, 0), text, font=self.font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            w, h = draw.textsize(text, font=self.font)

        padding = 6
        bg_rect = [0 - padding, 0 - padding, w + padding, h + padding]
        draw.rectangle(bg_rect, fill="black")
        draw.text((0, 0), text, font=self.font, fill="red")

        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

# ─── Start Webcam ───────────────────────────────────────────────────────────────
webrtc_ctx = webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTC_CONFIGURATION,   # <— add this
    video_processor_factory=StableProcessor,
    async_processing=True
)

st.markdown("### 🧠 Stable Prediction (after 3 consistent frames):")
st.success(st.session_state.get("stable_prediction", "Waiting..."))

# ─── NEW: Single Image Upload & Classification ──────────────────────────────────
st.markdown("---")
st.header("🖼️ Upload an Image for Classification")

uploaded = st.file_uploader(
    "Upload a Drosophila image (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded is not None:
    pil_in = Image.open(uploaded).convert("RGB")
    label, conf, probs = classify(pil_in)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Prediction")
        st.write(f"**Stage:** {label}")
        st.write(f"**Confidence:** {conf:.2%}")

    annotated = draw_label(pil_in, label, conf)
    with col2:
        st.subheader("Annotated Preview")
        st.image(annotated, use_column_width=True)

    with st.expander("Show class probabilities"):
        rows = [(lab, float(p)) for lab, p in zip(STAGE_LABELS, probs)]
        rows.sort(key=lambda x: x[1], reverse=True)
        for lab, p in rows:
            st.write(f"{lab}: {p:.2%}")
