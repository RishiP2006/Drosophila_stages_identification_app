# app.py
import time
import io
import streamlit as st
st.set_page_config(layout="centered")

# (Keep shim for some older wheels)
if not hasattr(st, "experimental_rerun") and hasattr(st, "rerun"):
    try:
        st.experimental_rerun = st.rerun  # type: ignore
    except Exception:
        pass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from huggingface_hub import hf_hub_download

# KERAS 3 runtime
from keras.models import load_model as k_load_model
from keras.applications.inception_v3 import preprocess_input

# ── Config ──────────────────────────────────────────────────────────────────────
HF_REPO_ID = "RishiPTrial/my-model-name"
MODEL_FILE = "drosophila_inceptionv3_classifier.h5"
INPUT_SIZE = 299
STAGE_LABELS = ["egg", "1st instar", "2nd instar", "3rd instar", "white pupa", "brown pupa", "eye pupa"]

# (Optional) WebRTC imports guarded so the app runs even if that stack isn’t installed/working
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    import av  # needed by streamlit_webrtc/aiortc
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False

# ── Load Model ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model from Hugging Face…")
def load_model():
    token = st.secrets.get("HF_TOKEN", None)
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE, token=token)
    return k_load_model(model_path, compile=False)

model = load_model()

# ── Helpers ─────────────────────────────────────────────────────────────────────
def preprocess_image(pil: Image.Image) -> np.ndarray:
    pil = pil.resize((INPUT_SIZE, INPUT_SIZE)).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32)
    return preprocess_input(arr)  # InceptionV3 scaling [-1, 1]

def classify(pil: Image.Image):
    arr = preprocess_image(pil)
    preds = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(preds))
    return STAGE_LABELS[idx], float(preds[idx]), preds

def annotate(pil: Image.Image, text: str) -> Image.Image:
    out = pil.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        w, h = draw.textsize(text, font=font)
    pad = 6
    draw.rectangle([0 - pad, 0 - pad, w + pad, h + pad], fill="black")
    draw.text((0, 0), text, font=font, fill="red")
    return out

def show_probs(labels, probs):
    rows = sorted([(lab, float(p)) for lab, p in zip(labels, probs)], key=lambda x: x[1], reverse=True)
    for lab, p in rows:
        st.write(f"{lab}: {p:.2%}")

# ── UI ──────────────────────────────────────────────────────────────────────────
st.title("Drosophila Stage Identification")

mode = st.radio(
    "Choose input mode",
    ["Upload", "Camera (simple)", "Live (beta)"],
    index=1,  # default to camera (simple) which works on Cloud
    help="If Live (beta) buffers/white screen on Cloud, use Camera (simple) or Upload."
)

st.divider()

# ── Mode: Upload ────────────────────────────────────────────────────────────────
if mode == "Upload":
    up = st.file_uploader("Upload a Drosophila image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if up is not None:
        pil = Image.open(up).convert("RGB")
        label, conf, probs = classify(pil)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prediction")
            st.write(f"**Stage:** {label}")
            st.write(f"**Confidence:** {conf:.2%}")
            with st.expander("Class probabilities"):
                show_probs(STAGE_LABELS, probs)
        with col2:
            st.subheader("Annotated")
            st.image(annotate(pil, f"{label} ({conf:.0%})"), use_column_width=True)

# ── Mode: Camera (simple) — uses Streamlit’s built-in camera widget ────────────
elif mode == "Camera (simple)":
    st.caption("Click **Take Photo** to run the model. This path avoids WebRTC and is reliable on Streamlit Cloud.")
    snap = st.camera_input("Camera")
    if snap:
        # Read bytes -> PIL
        pil = Image.open(io.BytesIO(snap.getvalue())).convert("RGB")
        label, conf, probs = classify(pil)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prediction")
            st.write(f"**Stage:** {label}")
            st.write(f"**Confidence:** {conf:.2%}")
            with st.expander("Class probabilities"):
                show_probs(STAGE_LABELS, probs)
        with col2:
            st.subheader("Annotated")
            st.image(annotate(pil, f"{label} ({conf:.0%})"), use_column_width=True)

# ── Mode: Live (beta) — keep optional, but not required for a working app ──────
elif mode == "Live (beta)":
    if not HAS_WEBRTC:
        st.warning("WebRTC stack not installed or failed to import on this build. Use 'Camera (simple)' or 'Upload'.")
    else:
        st.caption("If you see buffering/white screen on Cloud, your network blocks P2P. Try the **Force TURN** option below or switch modes.")
        # TURN/STUN config
        use_turn_only = st.checkbox("Force TURN relay only", value=True)
        RTC_CONFIGURATION = {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {
                    "urls": [
                        "turn:openrelay.metered.ca:80",
                        "turn:openrelay.metered.ca:443",
                        "turn:openrelay.metered.ca:443?transport=tcp",
                    ],
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
                },
            ],
            "iceTransportPolicy": "relay" if use_turn_only else "all",
        }
        MEDIA_STREAM_CONSTRAINTS = {
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 15}},
            "audio": False,
        }

        class Processor(VideoProcessorBase):
            def __init__(self):
                self.last_label = None
                self.count = 0
                self.font = ImageFont.load_default()
                try:
                    self.font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
                except Exception:
                    pass
                self.last_infer_t = 0.0
                self.infer_interval_s = 0.4

            def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
                img = frame.to_ndarray(format="rgb24")
                pil = Image.fromarray(img)
                now = time.time()
                text = "…"
                if now - self.last_infer_t >= self.infer_interval_s:
                    self.last_infer_t = now
                    label, conf, _ = classify(pil)
                    text = f"{label} ({conf:.0%})"
                elif self.last_label:
                    text = self.last_label
                out = annotate(pil, text)
                return av.VideoFrame.from_ndarray(np.array(out), format="rgb24")

        webrtc_streamer(
            key=f"live-{'relay' if use_turn_only else 'all'}",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=Processor,
            async_processing=False,  # fewer internal reruns on Cloud
            sendback_audio=False,
        )
