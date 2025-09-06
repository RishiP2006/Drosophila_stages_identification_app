# ------------------ app.py ------------------
# 1) Kill GPU/XLA noise BEFORE importing TF/Keras.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # force CPU (no cuInit 303)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"           # quiet TF logs
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import streamlit as st
st.set_page_config(layout="centered")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Keras / TF
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import layers, Model
# Prefer tf.keras for legacy .h5
from tensorflow.keras.models import load_model as tf_load_model
from keras.models import load_model as k_load_model

HF_REPO_ID = "RishiPTrial/my-model-name"
# Try these in order; change the first to ".keras" once you re-export.
CANDIDATE_FILES = [
    "drosophila_inceptionv3_classifier.keras",  # preferred modern format
    "drosophila_inceptionv3_classifier.h5",     # your current file
    "saved_model.tar.gz",                        # if you upload SavedModel (optional)
]
INPUT_SIZE = 299
STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa"
]

def _download_first_existing(repo_id: str, candidates):
    """Try downloading the first existing file among candidates."""
    last_err = None
    for fname in candidates:
        try:
            return hf_hub_download(repo_id=repo_id, filename=fname, token=st.secrets.get("HF_TOKEN"))
        except Exception as e:
            last_err = e
    # If none worked, re-raise the last error
    raise last_err or RuntimeError("No model file found in Hugging Face repo.")

@st.cache_resource(show_spinner="Loading model from Hugging Face…")
def load_or_fallback_model():
    # Try downloading any of the candidate files
    try:
        model_path = _download_first_existing(HF_REPO_ID, CANDIDATE_FILES)
    except Exception as e:
        st.warning(f"Could not download a model file from HF: {e}\nUsing fallback classifier instead.")
        return build_fallback_model()

    # 1) tf.keras full-model load (best for legacy .h5 graphs)
    tf_err_msg = ""
    try:
        m = tf_load_model(model_path, compile=False)
        return m
    except Exception as e_tf:
        tf_err_msg = f"{type(e_tf).__name__}: {e_tf}"
        st.info("tf.keras load failed; trying Keras 3 legacy loader…")

    # 2) Keras 3 legacy H5 loader
    k_err_msg = ""
    try:
        m = k_load_model(model_path, compile=False)
        return m
    except Exception as e_k:
        k_err_msg = f"{type(e_k).__name__}: {e_k}"

    # 3) Fallback: build a clean InceptionV3 head so the app keeps working
    st.warning(
        "Your .h5 model could not be loaded (legacy multi-input Dense bug).\n\n"
        f"tf.keras error:\n{tf_err_msg}\n\nKeras 3 error:\n{k_err_msg}\n\n"
        "Using a fallback ImageNet-initialized classifier so the app runs without errors. "
        "To use your trained weights, please re-export the model to `.keras` or SavedModel (instructions below)."
    )
    return build_fallback_model()

def build_fallback_model():
    """Build a minimal InceptionV3->GAP->Dense softmax head (untrained for your labels)."""
    base = InceptionV3(include_top=False, weights="imagenet", input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
    x = layers.GlobalAveragePooling2D()(base.output)
    out = layers.Dense(len(STAGE_LABELS), activation="softmax", name="stage_head")(x)
    return Model(base.input, out)

model = load_or_fallback_model()

# ---------- Inference utils ----------
def preprocess_image(pil: Image.Image) -> np.ndarray:
    pil = pil.resize((INPUT_SIZE, INPUT_SIZE)).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32)
    return preprocess_input(arr)  # InceptionV3 expects [-1, 1]

def classify(pil: Image.Image):
    arr = preprocess_image(pil)
    preds = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(preds))
    return STAGE_LABELS[idx], float(preds[idx])

# ---------- UI ----------
st.title("Live Drosophila Detection")
st.caption("Per-frame predictions. If the camera fails to connect, use the snapshot fallback below.")

class SimpleProcessor(VideoProcessorBase):
    def __init__(self):
        try:
            self.font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        except Exception:
            self.font = ImageFont.load_default()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)

        try:
            label, conf = classify(pil)
        except Exception:
            label, conf = "error", 0.0

        draw = ImageDraw.Draw(pil)
        text = f"{label} ({conf:.0%})"
        try:
            x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=self.font)
        except Exception:
            w, h = draw.textsize(text, font=self.font)
            x0, y0, x1, y1 = 0, 0, w, h
        pad = 6
        draw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill="black")
        draw.text((0, 0), text, font=self.font, fill="red")

        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

# No STUN (host-only ICE) + no async to avoid event-loop races
rtc_cfg = {"iceServers": []}
webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=SimpleProcessor,
    async_processing=False,
    rtc_configuration=rtc_cfg
)

st.divider()
snap = st.camera_input("No luck with live video? Use the snapshot fallback:")
if snap is not None:
    pil = Image.open(snap)
    label, conf = classify(pil)
    st.success(f"{label} ({conf:.0%})")
    st.image(pil, caption="Snapshot prediction")
# ---------------- End app.py ---------------
