import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download, HfApi
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import tensorflow as tf
import re

st.set_page_config(page_title="Drosophila Stage Detection", layout="centered")
st.title("🧬 Drosophila Stage Detection (Ensemble Mode)")
st.write("Live camera ensemble prediction using multiple stage classification models.")

HF_REPO_ID = "RishiPTrial/stage_modelv2"
STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa", "black pupa"
]

# Preprocessing map
PREPROCESS_MAP = {
    'inceptionv3': tf.keras.applications.inception_v3.preprocess_input,
    'convnext': tf.keras.applications.convnext.preprocess_input,
    'resnet50': tf.keras.applications.resnet50.preprocess_input,
}

@st.cache_data(show_spinner=False)
def list_models():
    try:
        files = HfApi().list_repo_files(repo_id=HF_REPO_ID)
        return [f for f in files if f.endswith(".h5") or f.endswith(".keras")]
    except:
        return []

@st.cache_resource(show_spinner=False)
def load_ensemble_models():
    model_files = list_models()
    ensemble = []
    for name in model_files:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=name)
        size = 299 if 'inceptionv3' in name.lower() else 224
        key = 'inceptionv3' if 'inceptionv3' in name.lower() else (
            'convnext' if 'convnext' in name.lower() else 'resnet50')
        preprocess = PREPROCESS_MAP[key]
        try:
            model = tf.keras.models.load_model(path, compile=False, custom_objects={'preprocess_input': preprocess})
            ensemble.append((model, preprocess, size))
        except Exception as e:
            st.warning(f"Skipping model {name}: {e}")
    return ensemble

ENSEMBLE_MODELS = load_ensemble_models()

class EnsembleProcessor(VideoProcessorBase):
    def __init__(self):
        self.models = ENSEMBLE_MODELS

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        votes = np.zeros(len(STAGE_LABELS))

        for model, pre_fn, size in self.models:
            arr = np.array(pil.resize((size, size))).astype(np.float32)
            arr = pre_fn(arr)
            pred = model.predict(arr[np.newaxis], verbose=0)[0]
            votes += pred

        idx = int(np.argmax(votes))
        label = STAGE_LABELS[idx]
        conf = votes[idx] / np.sum(votes)

        draw = ImageDraw.Draw(pil)
        draw.text((10, 10), f"{label} ({conf:.0%})", fill="lime")
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

# Start camera
st.markdown("---")
st.subheader("📸 Live Camera Ensemble Prediction")

if ENSEMBLE_MODELS:
    webrtc_streamer(
        key="ensemblecam",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": {"width": 320, "height": 240}, "audio": False},
        video_processor_factory=EnsembleProcessor,
        async_processing=True,
    )
else:
    st.error("No valid models loaded for ensemble.")

st.markdown("---")
st.write("**Note:** Only ensemble mode is supported. Models are loaded from Hugging Face.")
