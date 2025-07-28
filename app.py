import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# ----------------------------
# Basic Setup
# ----------------------------
st.set_page_config(page_title="ResNet Drosophila Detector", layout="centered")
st.title("🧬 Drosophila Stage Detection (ResNet Only)")

# ----------------------------
# Constants
# ----------------------------
HF_REPO_ID = "RishiPTrial/stage_modelv2"
RESNET_MODEL_NAME = "drosophila_stage_resnet50_finetuned_IIT.keras"
STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa", "black pupa"
]

# ----------------------------
# Load ResNet50 Model
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_resnet_model():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=RESNET_MODEL_NAME)
    preprocess_input = tf.keras.applications.resnet50.preprocess_input
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'preprocess_input': preprocess_input})
    return model, preprocess_input

model, preprocess_input = load_resnet_model()
model_size = 224  # ResNet input size

# ----------------------------
# Live Camera Prediction
# ----------------------------
st.subheader("📸 Live Camera Prediction (ResNet50)")

class ResNetProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.preprocess = preprocess_input
        self.size = model_size

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        print("Processing frame...")  # DEBUG: Check if this is running
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        arr = np.array(pil.resize((self.size, self.size))).astype(np.float32)
        arr = self.preprocess(arr)
        preds = self.model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
        idx = np.argmax(preds)
        label = f"{STAGE_LABELS[idx]} ({preds[idx]:.0%})"
        draw = ImageDraw.Draw(pil)
        draw.text((10, 10), label, fill="lime")
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

# Start WebRTC Camera
webrtc_streamer(
    key="resnet-live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=ResNetProcessor,
    async_processing=True,
    video_html_attrs={"playsinline": True},
)
