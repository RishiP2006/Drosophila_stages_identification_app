import sys
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import re
from huggingface_hub import hf_hub_download, HfApi
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

st.set_page_config(page_title="Drosophila Stage Detection", layout="centered")
st.title("🧬 Drosophila Stage Detection")
st.write("Select a model and upload an image or use live camera.")

# Constants
HF_REPO_ID = "RishiPTrial/stage_modelv2"
STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa", "black pupa"
]

# List and build model info
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

# Lazy load preprocess functions map
PREPROCESS_MAP = {
    'inceptionv3': __import__('tensorflow.keras.applications.inception_v3', fromlist=['preprocess_input']).preprocess_input,
    'convnext': __import__('tensorflow.keras.applications.convnext', fromlist=['preprocess_input']).preprocess_input,
    'resnet50': __import__('tensorflow.keras.applications.resnet50', fromlist=['preprocess_input']).preprocess_input,
}

# Load selected model (cached)
@st.cache_resource(show_spinner=False)
def load_model(name, size):
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=name)
    import tensorflow as tf
    # choose preprocess fn by name key
    key = 'inceptionv3' if 'inceptionv3' in name.lower() else ('convnext' if 'convnext' in name.lower() else 'resnet50')
    preprocess_fn = PREPROCESS_MAP[key]
    model = tf.keras.models.load_model(path, compile=False, custom_objects={'preprocess_input': preprocess_fn})
    return model, preprocess_fn, size

# UI model selector
model_name = st.selectbox("Select model:", list(MODELS_INFO.keys()))
selected_size = MODELS_INFO.get(model_name, 224)
model, preprocess_fn, model_size = load_model(model_name, selected_size)

# Image upload
st.subheader("📷 Upload Image")
file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if file:
    img = Image.open(file).convert("RGB")
    arr = np.array(img.resize((model_size,model_size))).astype(np.float32)
    arr = preprocess_fn(arr)
    preds = model.predict(arr[np.newaxis],verbose=0)[0]
    idx = np.argmax(preds)
    st.image(img, caption=f"{STAGE_LABELS[idx]} ({preds[idx]:.2f})")

st.markdown("---")
st.subheader("📸 Live Camera Detection")

# Video processor with lazy model already loaded above
class FastProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.pre_fn = preprocess_fn
        self.size = model_size

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        arr = np.array(pil.resize((self.size,self.size))).astype(np.float32)
        arr = self.pre_fn(arr)
        preds = self.model.predict(arr[np.newaxis],verbose=0)[0]
        idx = np.argmax(preds)
        label = STAGE_LABELS[idx]
        # draw simple text
        draw = ImageDraw.Draw(pil)
        draw.text((10,10), f"{label} ({preds[idx]:.0%})", fill="lime")
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

webrtc_streamer(
    key="fastcam",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": {"width":320,"height":240}, "audio": False},
    video_processor_factory=FastProcessor,
    async_processing=True,
)
