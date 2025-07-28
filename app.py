import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download, HfApi
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import re

st.set_page_config(page_title="Drosophila Stage Detection", layout="centered")
st.title("🧬 Drosophila Stage Detection (Ensemble Mode)")
st.write("Uses an ensemble of all available models to detect developmental stage via live camera.")

HF_REPO_ID = "RishiPTrial/stage_modelv2"
STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa", "black pupa"
]

@st.cache_data(show_spinner=False)
def list_hf_models():
    try:
        files = HfApi().list_repo_files(repo_id=HF_REPO_ID)
        return [f for f in files if f.lower().endswith(".h5")]
    except:
        return []

@st.cache_data(show_spinner=False)
def build_models_info():
    info = {}
    for fname in list_hf_models():
        size = 299 if "inceptionv3" in fname.lower() else 224
        info[fname] = size
    return info

MODELS_INFO = build_models_info()
if not MODELS_INFO:
    st.error(f"No .h5 models in {HF_REPO_ID}")

PREPROCESS_MAP = {
    'inceptionv3': __import__('tensorflow.keras.applications.inception_v3', fromlist=['preprocess_input']).preprocess_input,
    'convnext': __import__('tensorflow.keras.applications.convnext', fromlist=['preprocess_input']).preprocess_input,
    'resnet50': __import__('tensorflow.keras.applications.resnet50', fromlist=['preprocess_input']).preprocess_input,
}

@st.cache_resource(show_spinner=False)
def load_all_models():
    import tensorflow as tf
    models = []
    for name, size in MODELS_INFO.items():
        key = 'inceptionv3' if 'inceptionv3' in name.lower() else ('convnext' if 'convnext' in name.lower() else 'resnet50')
        pre_fn = PREPROCESS_MAP[key]
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=name)
        model = tf.keras.models.load_model(path, compile=False, custom_objects={'preprocess_input': pre_fn})
        models.append((model, pre_fn, size))
    return models

ENSEMBLE_MODELS = load_all_models()

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

        votes /= len(self.models)
        idx = np.argmax(votes)
        label = STAGE_LABELS[idx]
        draw = ImageDraw.Draw(pil)
        draw.text((10, 10), f"{label} ({votes[idx]:.0%})", fill="lime")
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

st.subheader("📸 Live Camera Stage Detection (Ensemble)")

webrtc_streamer(
    key="ensemblecam",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={
        "video": {"width": 160, "height": 120, "frameRate": {"ideal": 15, "max": 15}},
        "audio": False
    },
    video_processor_factory=EnsembleProcessor,
    async_processing=False,
    rtc_configuration={"iceServers": []},
)

st.markdown("---")
st.write("**Notes:**")
st.write("- Using all .h5 models as an ensemble from HF repo: RishiPTrial/stage_modelv2")
st.write("- Removed single model selection to prioritize faster load time and robustness.")
st.write("- Ensemble prediction averages model outputs for consistent accuracy.")
