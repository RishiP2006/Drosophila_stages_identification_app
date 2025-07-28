import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import tensorflow as tf
import re

# --- CONFIG ---
st.set_page_config(page_title="Drosophila Stage Detection", layout="centered")
st.title("🧬 Ensemble Drosophila Stage Detection")

HF_REPO_ID = "RishiPTrial/stage_modelv2"
STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa", "black pupa"
]

# --- Identify Preprocessing ---
PREPROCESS_MAP = {
    "inceptionv3": tf.keras.applications.inception_v3.preprocess_input,
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
    "convnext": tf.keras.applications.convnext.preprocess_input,
}

def detect_preprocess_key(filename):
    fname = filename.lower()
    if "inception" in fname:
        return "inceptionv3", 299
    elif "resnet" in fname:
        return "resnet50", 224
    elif "convnext" in fname:
        return "convnext", 224
    else:
        return "resnet50", 224  # default

# --- Load all models ---
@st.cache_resource(show_spinner=True)
def load_all_models():
    files = [
        "best_convnext_model_IIT.keras",
        "drosophila_inceptionv3_classifier.h5",
        "drosophila_stage_resnet50_finetuned_IIT.keras"
    ]
    models = []
    for fname in files:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=fname)
        key, size = detect_preprocess_key(fname)
        preprocess_fn = PREPROCESS_MAP[key]
        model = tf.keras.models.load_model(path, compile=False, custom_objects={"preprocess_input": preprocess_fn})
        models.append((model, preprocess_fn, size))
    return models

ALL_MODELS = load_all_models()

# --- Image Preprocessing ---
def preprocess_image(img, size, preprocess_fn):
    img = img.resize((size, size))
    arr = np.array(img).astype(np.float32)
    return preprocess_fn(arr)

# --- Ensemble Prediction ---
def ensemble_predict(pil_img):
    votes = np.zeros(len(STAGE_LABELS))
    for model, pre_fn, size in ALL_MODELS:
        arr = preprocess_image(pil_img, size, pre_fn)
        preds = model.predict(arr[np.newaxis], verbose=0)[0]
        votes += preds
    votes /= len(ALL_MODELS)
    idx = int(np.argmax(votes))
    return STAGE_LABELS[idx], votes[idx]

# --- Webcam Prediction ---
class EnsembleProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        label, prob = ensemble_predict(pil)
        draw = ImageDraw.Draw(pil)
        draw.text((10, 10), f"{label} ({prob:.0%})", fill="lime")
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

# --- App Layout ---
st.markdown("### 📸 Live Camera Prediction (Ensemble Only)")
webrtc_streamer(
    key="ensemblecam",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": {"width": 320, "height": 240}, "audio": False},
    video_processor_factory=EnsembleProcessor,
    async_processing=True,
)

st.markdown("---")
st.caption("Note: Camera uses ensemble of 3 models (`InceptionV3`, `ConvNeXt`, `ResNet50`).")
