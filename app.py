import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download, HfApi
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import threading

st.set_page_config(page_title="Drosophila Stage Detection", layout="centered")
st.title("🧬 Drosophila Stage Detection")
st.write("Live ensemble-based Drosophila stage classification")

HF_REPO_ID = "RishiPTrial/stage_modelv2"
STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa", "black pupa"
]

# --- Preprocessing functions ---
PREPROCESS_MAP = {
    'inceptionv3': tf.keras.applications.inception_v3.preprocess_input,
    'convnext': tf.keras.applications.convnext.preprocess_input,
    'resnet50': tf.keras.applications.resnet50.preprocess_input
}

def detect_pre_key(name):
    name = name.lower()
    if 'inceptionv3' in name: return 'inceptionv3', 299
    if 'convnext' in name: return 'convnext', 224
    return 'resnet50', 224

# --- Global models ---
ENSEMBLE_MODELS = []
_loaded = False

def load_models_bg():
    global ENSEMBLE_MODELS, _loaded
    files = HfApi().list_repo_files(repo_id=HF_REPO_ID)
    for fn in files:
        if not fn.lower().endswith((".h5", ".keras")): continue
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=fn)
        key, size = detect_pre_key(fn)
        pre_fn = PREPROCESS_MAP[key]
        try:
            m = tf.keras.models.load_model(path, compile=False, custom_objects={"preprocess_input": pre_fn})
            # warm-up call
            _ = m.predict(np.zeros((1, size, size, 3), dtype=np.float32), verbose=0)
            ENSEMBLE_MODELS.append((m, pre_fn, size))
        except Exception as e:
            print("Failed to load:", fn, e)
    _loaded = True

@st.cache_resource(show_spinner=False)
def trigger_loader():
    thread = threading.Thread(target=load_models_bg)
    thread.start()
    return True

trigger_loader()

# --- Live Camera Prediction ---
class EnsembleProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        label = "Loading models..."

        if _loaded and ENSEMBLE_MODELS:
            votes = np.zeros(len(STAGE_LABELS), np.float32)
            for model, pre_fn, sz in ENSEMBLE_MODELS:
                arr = np.array(pil.resize((sz, sz))).astype(np.float32)
                arr = pre_fn(arr)
                p = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
                votes += p
            votes /= len(ENSEMBLE_MODELS)
            idx = int(np.argmax(votes))
            label = f"{STAGE_LABELS[idx]} ({votes[idx]:.0%})"

        draw = ImageDraw.Draw(pil)
        draw.text((10, 10), label, fill="lime")
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

st.subheader("📸 Live Ensemble Camera Detection")
webrtc_streamer(
    key="ensemblecam",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": {"width": 320, "height": 240}, "audio": False},
    video_processor_factory=EnsembleProcessor,
    async_processing=True
)
