import streamlit as st
st.set_page_config(layout="centered")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model as lm

# ─── Preprocessing ──────────────────────
def basic_preprocess(pil_img, size):
    pil_img = pil_img.resize((size, size)).convert("RGB")
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    return arr

# ─── Config ─────────────────────────────
HF_REPO_ID = "RishiPTrial/stage_modelv2"
MODELS_INFO = {
    "InceptionV3": {"filename": "drosophila_inceptionv3_classifier.h5", "size": 299},
    "ConvNeXt": {"filename": "best_convnext_model_IIT.keras", "size": 224},
    "ResNet50": {"filename": "drosophila_stage_resnet50_finetuned_IIT.keras", "size": 224},
}
STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa", "black pupa"
]

@st.cache_resource
def load_models():
    models = {}
    for name, info in MODELS_INFO.items():
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=info["filename"])
        models[name] = (lm(path, compile=False), info["size"])
    return models

models = load_models()

# ─── Predict ────────────────────────────
def predict_single(pil_img, model, size):
    arr = basic_preprocess(pil_img, size)
    preds = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
    idx = int(np.argmax(preds))
    return STAGE_LABELS[idx], float(preds[idx])

# ─── UI Setup ───────────────────────────
st.title("🧬 Live Drosophila Stage Detection")
mode = st.radio("Choose Mode", ["Single Model", "Ensemble"])

if mode == "Single Model":
    selected_model = st.selectbox("Choose a model", list(models.keys()))
    active_model = models[selected_model]
else:
    st.warning("Live prediction supports only Single Model mode for performance.")
    selected_model = "InceptionV3"
    active_model = models[selected_model]

# ─── Font Cache ─────────────────────────
@st.cache_resource
def get_font():
    return ImageFont.truetype("DejaVuSans-Bold.ttf", 28)

font = get_font()

# ─── Live Video ─────────────────────────
class FastVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model, self.size = active_model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)

        label, conf = predict_single(pil, self.model, self.size)

        draw = ImageDraw.Draw(pil)
        text = f"{label} ({conf:.0%})"
        text_size = draw.textbbox((0, 0), text, font=font)
        draw.rectangle(
            [text_size[0]-6, text_size[1]-6, text_size[2]+6, text_size[3]+6],
            fill="black"
        )
        draw.text((0, 0), text, font=font, fill="lime")

        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

st.subheader("📸 Live Camera Detection")
webrtc_streamer(
    key="drosophila-live-fast",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=FastVideoProcessor,
    async_processing=True
)
