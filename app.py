# app.py
import time
import streamlit as st
st.set_page_config(layout="centered")

# --- Compat shim: some streamlit-webrtc versions call st.experimental_rerun
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

# KERAS 3 runtime
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

# ✅ Use TURN (relay) so WebRTC works on Cloud behind NAT/firewalls
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
    ]
}

# Keep video lightweight to avoid timeouts/reloads
MEDIA_STREAM_CONSTRAINTS = {
    "video": {
        "width": {"ideal": 640},
        "height": {"ideal": 480},
        "frameRate": {"ideal": 15},
    },
    "audio": False,
}

# ─── Load Model (Keras 3) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model from Hugging Face…")
def load_model():
    token = st.secrets.get("HF_TOKEN", None)
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE, token=token)
    return k_load_model(model_path, compile=False)

model = load_model()

# ─── Helpers ────────────────────────────────────────────────────────────────────
def preprocess_image(pil: Image.Image) -> np.ndarray:
    pil = pil.resize((INPUT_SIZE, INPUT_SIZE)).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32)
    return preprocess_input(arr)  # InceptionV3 scaling [-1, 1]

def classify(pil: Image.Image):
    arr = preprocess_image(pil)
    preds = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(preds))
    return STAGE_LABELS[idx], float(preds[idx]), preds

def draw_legend(pil: Image.Image, text: str, font):
    out = pil.copy()
    draw = ImageDraw.Draw(out)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        w, h = draw.textsize(text, font=font)
    padding = 6
    bg = [0 - padding, 0 - padding, w + padding, h + padding]
    draw.rectangle(bg, fill="black")
    draw.text((0, 0), text, font=font, fill="red")
    return out

# ─── UI ─────────────────────────────────────────────────────────────────────────
st.title("Live Drosophila Detection")
st.caption("Tip: if camera stays white, check browser camera permission and try another browser tab.")

# ─── Video Processor (no Streamlit calls here) ──────────────────────────────────
class StableProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_label = None
        self.count = 0
        self.stable_label = None
        self.stable_conf = 0.0

        # throttle inference (once every 0.4 s)
        self.last_infer_t = 0.0
        self.infer_interval_s = 0.4

        try:
            self.font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        except Exception:
            self.font = ImageFont.load_default()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)

        t = time.time()
        label_text = "…"

        if (t - self.last_infer_t) >= self.infer_interval_s:
            self.last_infer_t = t
            label, conf, _ = classify(pil)

            if label == self.last_label:
                self.count += 1
            else:
                self.last_label = label
                self.count = 1

            if self.count >= 3:
                self.stable_label = label
                self.stable_conf = conf

            label_text = f"{label} ({conf:.0%})"
        else:
            if self.last_label:
                label_text = f"{self.last_label}"

        annotated = draw_legend(pil, label_text, self.font)
        return av.VideoFrame.from_ndarray(np.array(annotated), format="rgb24")

# ─── Start Webcam (sync processing, TURN, small video) ──────────────────────────
webrtc_ctx = webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=StableProcessor,
    async_processing=False,   # <= fewer internal reruns; more stable on Cloud
    sendback_audio=False,
)

# ─── Upload Image Path (unchanged) ──────────────────────────────────────────────
st.markdown("---")
st.header("🖼️ Upload an Image for Classification")

uploaded = st.file_uploader(
    "Upload a Drosophila image (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if uploaded is not None:
    pil_in = Image.open(uploaded).convert("RGB")
    label, conf, probs = classify(pil_in)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Prediction")
        st.write(f"**Stage:** {label}")
        st.write(f"**Confidence:** {conf:.2%}")

    annotated = draw_legend(pil_in, f"{label} ({conf:.0%})", ImageFont.load_default())
    with c2:
        st.subheader("Annotated Preview")
        st.image(annotated, use_column_width=True)

    with st.expander("Show class probabilities"):
        rows = sorted([(lab, float(p)) for lab, p in zip(STAGE_LABELS, probs)],
                      key=lambda x: x[1], reverse=True)
        for lab, p in rows:
            st.write(f"{lab}: {p:.2%}")
