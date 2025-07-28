import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download, HfApi
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av
import tensorflow as tf
import re

st.set_page_config(page_title="Drosophila Stage Detection", layout="centered")
st.title("🧬 Drosophila Stage Detection (Ensemble Only)")
st.write("Live prediction using ensemble of all uploaded models.")

# Constants
HF_REPO_ID = "RishiPTrial/stage_modelv2"
STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa", "black pupa"
]

# Cached Hugging Face .h5 or .keras models
@st.cache_data(show_spinner=False)
def list_models():
    try:
        files = HfApi().list_repo_files(repo_id=HF_REPO_ID)
        return [f for f in files if f.endswith((".h5", ".keras")) and not f.startswith(".")]
    except Exception:
        return []

@st.cache_resource(show_spinner=False)
def load_all_models():
    files = list_models()
    models = []
    for fname in files:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=fname)
        key = 'inceptionv3' if 'inceptionv3' in fname.lower() else (
              'convnext' if 'convnext' in fname.lower() else 'resnet50')
        preprocess_fn = __import__(f"tensorflow.keras.applications.{key}", fromlist=["preprocess_input"]).preprocess_input
        size = 299 if key == 'inceptionv3' else 224
        try:
            model = tf.keras.models.load_model(path, compile=False, custom_objects={'preprocess_input': preprocess_fn})
            models.append((model, preprocess_fn, size))
        except Exception as e:
            st.warning(f"Skipping {fname} due to error: {e}")
    return models

ALL_MODELS = load_all_models()
if not ALL_MODELS:
    st.error("No valid models found in Hugging Face repo.")
    st.stop()

# Live processor
class EnsembleProcessor(VideoProcessorBase):
    def __init__(self):
        self.models = ALL_MODELS

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        final_preds = np.zeros(len(STAGE_LABELS))

        for model, pre_fn, size in self.models:
            arr = np.array(pil.resize((size, size))).astype(np.float32)
            arr = pre_fn(arr)
            pred = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
            final_preds += pred

        final_preds /= len(self.models)
        idx = int(np.argmax(final_preds))
        label = f"{STAGE_LABELS[idx]} ({final_preds[idx]:.0%})"

        draw = ImageDraw.Draw(pil)
        draw.text((10, 10), label, fill="lime")
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

# Faster RTC config and low resolution
rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Live camera UI
st.markdown("---")
st.subheader("📸 Live Ensemble Stage Detection")

webrtc_streamer(
    key="ensemblecam",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": {"width": 320, "height": 240}, "audio": False},
    video_processor_factory=EnsembleProcessor,
    async_processing=True
)

st.markdown("---")
st.write("**Notes:**")
st.write(f"- Ensemble runs all models from Hugging Face repo `{HF_REPO_ID}`")
st.write("- Only `.h5` or `.keras` classification models supported")
st.write("- Camera resolution reduced for faster loading")
