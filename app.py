import threading
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download, HfApi
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import tensorflow as tf

st.set_page_config(page_title="Drosophila Stage Detection", layout="centered")
st.title("🧬 Ensemble Drosophila Stage Detection")
st.write("Live camera starts immediately; ensemble models load in background.")

HF_REPO_ID = "RishiPTrial/stage_modelv2"
STAGE_LABELS = [
    "egg","1st instar","2nd instar","3rd instar",
    "white pupa","brown pupa","eye pupa","black pupa"
]

# Map for preprocessing
PREPROCESS_MAP = {
    "inceptionv3": tf.keras.applications.inception_v3.preprocess_input,
    "resnet50":    tf.keras.applications.resnet50.preprocess_input,
    "convnext":    tf.keras.applications.convnext.preprocess_input,
}

def detect_pre_key(name):
    n = name.lower()
    if "inception" in n: return "inceptionv3",299
    if "convnext"   in n: return "convnext",224
    if "resnet"     in n: return "resnet50",224
    return "resnet50",224

# Global shared models list
ENSEMBLE_MODELS = []
_loaded = False

# Background loading
def load_models_bg():
    global ENSEMBLE_MODELS, _loaded
    files = HfApi().list_repo_files(repo_id=HF_REPO_ID)
    for fn in files:
        if not fn.lower().endswith((".h5",".keras")): continue
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=fn)
        key, size = detect_pre_key(fn)
        pre_fn = PREPROCESS_MAP[key]
        try:
            m = tf.keras.models.load_model(path, compile=False,
                                          custom_objects={"preprocess_input":pre_fn})
            # warm up
            _ = m.predict(np.zeros((1,size,size,3)),verbose=0)
            ENSEMBLE_MODELS.append((m,pre_fn,size))
        except Exception:
            pass
    _loaded = True

threading.Thread(target=load_models_bg, daemon=True).start()

# Dummy processor: just pass frames through
class PreviewProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        return frame

# Real ensemble processor
class EnsembleProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        # do ensemble only when loaded
        if _loaded and ENSEMBLE_MODELS:
            votes = np.zeros(len(STAGE_LABELS))
            for m,fn,sz in ENSEMBLE_MODELS:
                arr = np.array(pil.resize((sz,sz))).astype(np.float32)
                arr = fn(arr)
                p = m.predict(np.expand_dims(arr,0),verbose=0)[0]
                votes += p
            votes /= len(ENSEMBLE_MODELS)
            idx = int(np.argmax(votes))
            label = f"{STAGE_LABELS[idx]} ({votes[idx]:.0%})"
        else:
            label = "Loading models..."
        draw = ImageDraw.Draw(pil)
        draw.text((10,10), label, fill="lime")
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

# Launch camera immediately with preview, then switch to ensemble once ready
webrtc_ctx = webrtc_streamer(
    key="livecam",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video":{"width":320,"height":240},"audio":False},
    video_processor_factory=PreviewProcessor,
    async_processing=True
)

# Poll in the background for when models are ready, then swap
def switch_processor():
    import time
    while not _loaded:
        time.sleep(0.5)
    # swap to EnsembleProcessor
    webrtc_ctx.video_processor_factory = EnsembleProcessor

threading.Thread(target=switch_processor, daemon=True).start()

st.markdown("---")
st.write("⏳ Starting camera instantly. Models load async; ensemble predictions overlay when ready.")
