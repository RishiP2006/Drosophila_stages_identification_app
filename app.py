# app.py
import time
import streamlit as st
st.set_page_config(layout="centered")

# --- Compat shim: some streamlit-webrtc versions call st.experimental_rerun
if not hasattr(st, "experimental_rerun") and hasattr(st, "rerun"):
    st.experimental_rerun = st.rerun  # type: ignore[attr-defined]

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
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# ─── Load Model (Keras 3) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model from Hugging Face…")
def load_model():
    token = st.secrets.get("HF_TOKEN", None)  # if HF repo is private
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE, token=token)
    return k_load_model(model_path, compile=False)

model = load_model()

# ─── Image Preprocessing & Prediction ───────────────────────────────────────────
def preprocess_image(pil: Image.Image) -> np.ndarray:
    pil = pil.resize((INPUT_SIZE, INPUT_SIZE)).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32)
    return preprocess_input(arr)  # InceptionV3 scaling [-1, 1]

def classify(pil: Image.Image):
    arr = preprocess_image(pil)
    preds = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(preds))
    return STAGE_LABELS[idx], float(preds[idx]), preds

def draw_label(pil: Image.Image, text: str, fill: str = "red"):
    pil = pil.copy()
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        w, h = draw.textsize(text, font=font)
    padding = 6
    bg_rect = [0 - padding, 0 - padding, w + padding, h + padding]
    draw.rectangle(bg_rect, fill="black")
    draw.text((0, 0), text, font=font, fill=fill)
    return pil

# ─── UI ─────────────────────────────────────────────────────────────────────────
st.title("Live Drosophila Detection")
st.subheader("📹 Live Camera Detection with Stable Prediction")

# place where we show the stable prediction (main thread only)
stable_pred_box = st.empty()

# ─── Video Processor ────────────────────────────────────────────────────────────
class StableProcessor(VideoProcessorBase):
    def __init__(self):
        # state used only within this worker thread
        self.last_label = None
        self.count = 0
        self.stable_label = None
        self.stable_conf = 0.0

        # throttle inference (once every 0.3 s)
        self.last_infer_t = 0.0
        self.infer_interval_s = 0.3

        try:
            self.font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        except Exception:
            self.font = ImageFont.load_default()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)

        # throttle heavy inference
        t = time.time()
        do_infer = (t - self.last_infer_t) >= self.infer_interval_s
        if do_infer:
            self.last_infer_t = t
            label, conf, _ = classify(pil)

            # stability over consecutive inferences (not per-frame)
            if label == self.last_label:
                self.count += 1
            else:
                self.last_label = label
                self.count = 1

            if self.count >= 3:
                self.stable_label = label
                self.stable_conf = conf

            text = f"{label} ({conf:.0%})"
        else:
            # draw last seen label if available
            text = f"{self.last_label or '…'}"

        # overlay current label (not touching session_state here)
        try:
            bbox = ImageDraw.Draw(pil).textbbox((0, 0), text, font=self.font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            w, h = ImageDraw.Draw(pil).textsize(text, font=self.font)
        padding = 6
        bg_rect = [0 - padding, 0 - padding, w + padding, h + padding]
        draw = ImageDraw.Draw(pil)
        draw.rectangle(bg_rect, fill="black")
        draw.text((0, 0), text, font=self.font, fill="red")

        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

# ─── Start Webcam ───────────────────────────────────────────────────────────────
webrtc_ctx = webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=StableProcessor,
    async_processing=True,
)

# Show stable prediction from the main thread without mutating state in recv
if webrtc_ctx and webrtc_ctx.state.playing:
    # auto-refresh this small area every 700 ms while playing
    st.autorefresh(interval=700, key="pred_refresh")
    vp = webrtc_ctx.video_processor
    if vp and vp.stable_label:
        stable_pred_box.success(f"🧠 Stable Prediction: **{vp.stable_label}** ({vp.stable_conf:.2%})")
    else:
        stable_pred_box.info("🧠 Stable Prediction: gathering…")

# ─── Single Image Upload & Classification ───────────────────────────────────────
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

    annotated = draw_label(pil_in, f"{label} ({conf:.0%})")
    with col2:
        st.subheader("Annotated Preview")
        st.image(annotated, use_column_width=True)

    with st.expander("Show class probabilities"):
        rows = sorted(
            [(lab, float(p)) for lab, p in zip(STAGE_LABELS, probs)],
            key=lambda x: x[1],
            reverse=True,
        )
        for lab, p in rows:
            st.write(f"{lab}: {p:.2%}")
