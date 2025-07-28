import sys
import threading
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download, HfApi
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

st.set_page_config(page_title="Drosophila Stage Detection", layout="centered")
st.title("🧬 Drosophila Stage Detection")
st.write("Ensemble mode only: live camera uses all models together.")

# Constants
HF_REPO_ID = "RishiPTrial/stage_modelv2"
STAGE_LABELS = [
    "egg","1st instar","2nd instar","3rd instar",
    "white pupa","brown pupa","eye pupa","black pupa"
]

# Find .keras models
@st.cache_data(show_spinner=False)
def list_hf_models():
    try:
        files = HfApi().list_repo_files(repo_id=HF_REPO_ID)
        return [f for f in files if f.lower().endswith(".keras")]
    except:
        return []

MODEL_NAMES = list_hf_models()
if not MODEL_NAMES:
    st.error(f"No .keras models found in {HF_REPO_ID}")

# Preprocessing lookup
PREPROCESS_MAP = {
    'inceptionv3': __import__('tensorflow.keras.applications.inception_v3', fromlist=['preprocess_input']).preprocess_input,
    'convnext':   __import__('tensorflow.keras.applications.convnext',   fromlist=['preprocess_input']).preprocess_input,
    'resnet50':   __import__('tensorflow.keras.applications.resnet50',   fromlist=['preprocess_input']).preprocess_input,
}

# Global store for loaded models
models = {}
_state = {'counter': 0, 'label': 'Loading models...'}

def load_all_models():
    import tensorflow as tf
    for name in MODEL_NAMES:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=name)
        key = (
            'inceptionv3' if 'inceptionv3' in name.lower() else
            ('convnext' if 'convnext' in name.lower() else 'resnet50')
        )
        pre_fn = PREPROCESS_MAP[key]
        size = 299 if key=='inceptionv3' else 224
        m = tf.keras.models.load_model(path, compile=False, custom_objects={'preprocess_input': pre_fn})
        # warm‑up
        _ = m.predict(np.zeros((1, size, size, 3), dtype=np.float32), verbose=0)
        models[name] = (m, pre_fn, size)

# Start loading in background
threading.Thread(target=load_all_models, daemon=True).start()

# Upload + ensemble for static images
st.subheader("📷 Upload Image (Ensemble)")
file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if file and models:
    img = Image.open(file).convert("RGB")
    votes, confs = [], {}
    for m, fn, sz in models.values():
        arr = np.array(img.resize((sz,sz))).astype(np.float32)
        arr = fn(arr)
        p = m.predict(arr[np.newaxis], verbose=0)[0]
        idx = np.argmax(p)
        lbl = STAGE_LABELS[idx]
        votes.append(lbl)
        confs.setdefault(lbl, []).append(p[idx])
    from collections import Counter
    top = Counter(votes).most_common()
    if len(top)>1 and top[0][1]==top[1][1]:
        avg = {l:np.mean(v) for l,v in confs.items()}
        choice = max(avg, key=avg.get)
    else:
        choice = top[0][0]
    best = max(confs[choice])
    st.image(img, caption=f"{choice} ({best:.2f})")

st.markdown("---")
st.subheader("📸 Live Camera Ensemble Detection")
st.write("Camera opens immediately; predictions every 5 frames.")

class EnsembleProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        _state['counter'] += 1

        if models and _state['counter'] % 5 == 0:
            votes, confs = [], {}
            for m, fn, sz in models.values():
                arr = np.array(pil.resize((sz,sz))).astype(np.float32)
                arr = fn(arr)
                p = m.predict(arr[np.newaxis], verbose=0)[0]
                idx = np.argmax(p)
                lbl = STAGE_LABELS[idx]
                votes.append(lbl)
                confs.setdefault(lbl, []).append(p[idx])
            from collections import Counter
            top = Counter(votes).most_common()
            if len(top)>1 and top[0][1]==top[1][1]:
                avg = {l:np.mean(v) for l,v in confs.items()}
                choice = max(avg, key=avg.get)
            else:
                choice = top[0][0]
            _state['label'] = f"{choice} ({max(confs[choice]):.0%})"

        draw = ImageDraw.Draw(pil)
        draw.text((10,10), _state['label'], fill="lime")
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

webrtc_streamer(
    key="ensemblecam",
    mode=WebRtcMode.SENDONLY,  # send video to server only—no RTX decoding
    media_stream_constraints={
        "video": {
            "width": 160, "height": 120,
            "frameRate": {"ideal": 15, "max": 15}
        },
        "audio": False
    },
    video_processor_factory=EnsembleProcessor,
    async_processing=False,   # run inference in main thread to avoid asyncio errors
)
