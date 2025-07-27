import streamlit as st
st.set_page_config(layout="centered")

import numpy as np
from PIL import Image, ImageDraw
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model as lm
from tensorflow.keras.applications.inception_v3 import preprocess_input as iv3_preprocess
from tensorflow.keras.applications.convnext import preprocess_input as cx_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# ─── Config ─────────────────────────────
HF_REPO_ID = "RishiPTrial/stage_modelv2"
MODELS_INFO = {
    "InceptionV3": {"filename": "drosophila_inceptionv3_classifier.h5", "preprocess": iv3_preprocess, "size": 299},
    "ConvNeXt": {"filename": "best_convnext_model_IIT.keras", "preprocess": cx_preprocess, "size": 224},
    "ResNet50": {"filename": "drosophila_stage_resnet50_finetuned_IIT.keras", "preprocess": resnet_preprocess, "size": 224},
}
STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa", "black pupa"
]

# ─── Load Models ─────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for name, info in MODELS_INFO.items():
        try:
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=info["filename"])
            model = lm(path, compile=False, custom_objects={"preprocess_input": info["preprocess"]})
            models[name] = (model, info["preprocess"], info["size"])
        except Exception as e:
            st.warning(f"Failed to load model {name}: {e}")
    return models

models = load_models()

# ─── Prediction Utilities ────────────────
def preprocess_image(pil: Image.Image, size: int, preprocess_fn):
    pil = pil.resize((size, size)).convert("RGB")
    arr = np.asarray(pil, np.float32)
    return preprocess_fn(arr)

def classify_single(pil: Image.Image, model, preprocess_fn, size: int):
    arr = preprocess_image(pil, size, preprocess_fn)
    preds = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(preds))
    return STAGE_LABELS[idx], float(preds[idx])

def classify_ensemble(pil: Image.Image, models_dict):
    votes = []
    confidences = {}
    for _, (model, pre_fn, size) in models_dict.items():
        label, conf = classify_single(pil, model, pre_fn, size)
        votes.append(label)
        confidences.setdefault(label, []).append(conf)

    from collections import Counter
    count = Counter(votes)
    top_votes = count.most_common()
    if len(top_votes) > 1 and top_votes[0][1] == top_votes[1][1]:
        avg_conf = {lbl: np.mean(confs) for lbl, confs in confidences.items()}
        chosen = max(avg_conf, key=avg_conf.get)
    else:
        chosen = top_votes[0][0]

    return chosen, max(confidences[chosen])

# ─── UI ──────────────────────────────────
st.title("🧬 Live Drosophila Stage Detection")
mode = st.radio("Choose Mode", ["Single Model", "Ensemble"])

if mode == "Single Model":
    selected_model = st.selectbox("Choose a model", list(models.keys()))
    single_model_cfg = models[selected_model]
else:
    single_model_cfg = None
    st.info("Using all models in ensemble")

# ─── Fast Video Processor ────────────────
class FastProcessor(VideoProcessorBase):
    def __init__(self):
        self.mode = mode
        self.single_model_cfg = single_model_cfg
        self.models = models

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)

        if self.mode == "Single Model":
            label, conf = classify_single(pil, *self.single_model_cfg)
        else:
            label, conf = classify_ensemble(pil, self.models)

        # Fast drawing (no font load)
        draw = ImageDraw.Draw(pil)
        draw.text((10, 10), f"{label} ({conf:.0%})", fill="lime")

        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

# ─── Live Camera ─────────────────────────
st.subheader("📸 Live Camera Detection")

webrtc_streamer(
    key="fast-live-drosophila",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={
        "video": {"width": {"ideal": 320}, "height": {"ideal": 240}},
        "audio": False
    },
    video_processor_factory=FastProcessor,
    async_processing=True,
)
