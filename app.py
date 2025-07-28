import streamlit as st
st.set_page_config(layout="centered")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model as lm
from tensorflow.keras.applications.inception_v3 import preprocess_input

# ─── Model Info ───────────────────────────
HF_REPO_ID = "RishiPTrial/stage_modelv2"
MODEL_FILE = "drosophila_inceptionv3_classifier.h5"
STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa", "black pupa"
]

# ─── Load Model ───────────────────────────
@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)
    return lm(path, compile=False), 299

with st.spinner("🔄 Loading model..."):
    model, input_size = load_model()

# ─── Image Preprocessing ─────────────────
def preprocess_image(pil: Image.Image):
    pil = pil.resize((input_size, input_size)).convert("RGB")
    arr = np.asarray(pil, np.float32)
    arr = preprocess_input(arr)
    return arr

# ─── Prediction ──────────────────────────
def classify(pil: Image.Image):
    arr = preprocess_image(pil)
    preds = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
    idx = int(np.argmax(preds))
    return STAGE_LABELS[idx], float(preds[idx])

# ─── UI Setup ─────────────────────────────
st.title("🧬 Live Drosophila Stage Detection")
st.subheader("📹 Real-Time Camera Classification")

# ─── Session State ───────────────────────
st.session_state.setdefault("stable_prediction", "Waiting...")

# ─── WebRTC Configuration (Fix video/rtx) ─
rtc_configuration = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
    "sdpSemantics": "unified-plan",
}

# ─── Video Processor ─────────────────────
class StableProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_label = None
        self.count = 0
        self.stable_label = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)

        label, conf = classify(pil)

        # Stability check
        if label == self.last_label:
            self.count += 1
        else:
            self.last_label = label
            self.count = 1

        # Set stable label after 3 consistent predictions
        if self.count >= 3:
            self.stable_label = label
            st.session_state["stable_prediction"] = self.stable_label

        # Draw label
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        except IOError:
            font = ImageFont.load_default()

        text = f"{label} ({conf:.0%})" if conf > 0.5 else "Uncertain"
        text_size = draw.textbbox((0, 0), text, font=font)
        padding = 6
        bg_rect = [
            text_size[0] - padding,
            text_size[1] - padding,
            text_size[2] + padding,
            text_size[3] + padding
        ]
        draw.rectangle(bg_rect, fill="black")
        draw.text((0, 0), text, font=font, fill="red")

        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

# ─── Start Webcam ────────────────────────
webrtc_ctx = webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=rtc_configuration,  # 👈 Fix for `video/rtx` error
    video_processor_factory=StableProcessor,
    async_processing=True
)

# ─── Display Stable Prediction ───────────
st.markdown("### 🧠 Stable Prediction (after 3 consistent frames):")
st.success(st.session_state.get("stable_prediction", "Waiting..."))
