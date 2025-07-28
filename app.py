import sys
import threading
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

# Discover .keras models in repo
@st.cache_data(show_spinner=False)
def list_hf_models():
    try:
        files = HfApi().list_repo_files(repo_id=HF_REPO_ID)
        return [f for f in files if f.lower().endswith(".keras")]
    except:
        return []

MODELS = list_hf_models()
if not MODELS:
    st.error(f"No .keras models found in {HF_REPO_ID}")

model_name = st.selectbox("Select model:", MODELS)

# Preprocess map
PREPROCESS_MAP = {
    'inceptionv3': __import__('tensorflow.keras.applications.inception_v3', fromlist=['preprocess_input']).preprocess_input,
    'convnext':   __import__('tensorflow.keras.applications.convnext',   fromlist=['preprocess_input']).preprocess_input,
    'resnet50':   __import__('tensorflow.keras.applications.resnet50',   fromlist=['preprocess_input']).preprocess_input,
}

# Globals for background-loaded model
model = None
pre_fn = None
model_size = None

# Background loader
def load_model_bg(name):
    global model, pre_fn, model_size
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=name)
    import tensorflow as tf
    key = 'inceptionv3' if 'inceptionv3' in name.lower() else ('convnext' if 'convnext' in name.lower() else 'resnet50')
    pre_fn = PREPROCESS_MAP[key]
    # determine size from name suffix or default
    model_size = 299 if key=='inceptionv3' else 224
    # load and warm up
    m = tf.keras.models.load_model(path, compile=False, custom_objects={'preprocess_input': pre_fn})
    _ = m.predict(np.zeros((1, model_size, model_size,3), dtype=np.float32), verbose=0)
    model = m

# Kick off background load
threading.Thread(target=load_model_bg, args=(model_name,), daemon=True).start()

# Simple image uploader classification
st.subheader("📷 Upload Image")
file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if file and model is not None:
    img = Image.open(file).convert("RGB")
    arr = np.array(img.resize((model_size,model_size))).astype(np.float32)
    arr = pre_fn(arr)
    preds = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = np.argmax(preds)
    st.image(img, caption=f"{STAGE_LABELS[idx]} ({preds[idx]:.2f})")

st.markdown("---")
st.subheader("📸 Live Camera Detection")
st.write("(Camera preview opens immediately; predictions appear once model is ready)")

# Processor with frame skipping
defute = {'counter':0, 'last':f"Waiting..."}
class FastProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        arr_frame = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(arr_frame)
        # only predict every 5th frame and if model ready
        de = de
        defute['counter'] += 1
        if model is not None and defute['counter'] % 5 == 0:
            img_arr = np.array(pil.resize((model_size,model_size))).astype(np.float32)
            img_arr = pre_fn(img_arr)
            preds = model.predict(img_arr[np.newaxis],verbose=0)[0]
            idx = np.argmax(preds)
            defute['last'] = f"{STAGE_LABELS[idx]} ({preds[idx]:.0%})"
        # draw label
        draw = ImageDraw.Draw(pil)
        draw.text((10,10), defute['last'], fill="lime")
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

webrtc_streamer(
    key="fastcam",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={
        "video": {"width":160, "height":120, "frameRate":{"ideal":15, "max":15}},
        "audio": False
    },
    video_processor_factory=FastProcessor,
    async_processing=True
)
