# app.py
import streamlit as st
st.set_page_config(layout="centered")

import json
import numpy as np
import h5py
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model as tfk_load_model, model_from_json
from tensorflow.keras.applications.inception_v3 import preprocess_input

# ─── Config ─────────────────────────────────────────────────────────────────────
HF_REPO_ID = "RishiPTrial/my-model-name"
MODEL_FILE = "drosophila_inceptionv3_classifier.h5"
INPUT_SIZE = 299
STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa"
]

# ─── Helpers ────────────────────────────────────────────────────────────────────
def _patch_config_batch_shape_to_batch_input_shape(cfg: dict) -> dict:
    """
    Recursively walk a Keras model config dict and rename 'batch_shape' -> 'batch_input_shape'
    in any layer configs (esp. InputLayer). Returns a modified copy.
    """
    def fix_layer(layer):
        if isinstance(layer, dict):
            # If it's a Keras layer config
            if "class_name" in layer and "config" in layer and isinstance(layer["config"], dict):
                if "batch_shape" in layer["config"] and "batch_input_shape" not in layer["config"]:
                    layer["config"]["batch_input_shape"] = layer["config"].pop("batch_shape")
            # Recurse into nested structures
            for k, v in list(layer.items()):
                if isinstance(v, dict):
                    layer[k] = fix_layer(v)
                elif isinstance(v, list):
                    layer[k] = [fix_layer(x) if isinstance(x, dict) else x for x in v]
        return layer

    cfg = json.loads(json.dumps(cfg))  # deep copy
    return fix_layer(cfg)

def _load_model_with_patch(h5_path: str):
    """
    Fallback loader:
      - Read the JSON config from the H5
      - Patch 'batch_shape' -> 'batch_input_shape'
      - Rebuild model from JSON
      - Load weights from the same H5
    """
    with h5py.File(h5_path, "r") as f:
        # Keras stores the architecture JSON in the model_config attribute
        model_config_json = f.attrs.get("model_config")
        if model_config_json is None:
            raise ValueError("H5 file has no 'model_config' attribute.")
        if isinstance(model_config_json, bytes):
            model_config_json = model_config_json.decode("utf-8")

        cfg = json.loads(model_config_json)
        cfg_patched = _patch_config_batch_shape_to_batch_input_shape(cfg)
        patched_json = json.dumps(cfg_patched)

    model = model_from_json(patched_json)
    # Load weights from the same file
    model.load_weights(h5_path)
    return model

# ─── Cache + Load Model ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model from Hugging Face…")
def load_model():
    token = st.secrets.get("HF_TOKEN", None)
    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILE,
            token=token
        )
    except Exception as e:
        st.error(f"Could not download model file '{MODEL_FILE}' from '{HF_REPO_ID}': {e}")
        st.stop()

    # 1) Try the normal loader first
    try:
        return tfk_load_model(model_path, compile=False)
    except Exception as e:
        msg = str(e)
        # 2) If it failed due to the 'batch_shape' InputLayer issue, use the patch loader
        if "batch_shape" in msg and "InputLayer" in msg:
            try:
                patched = _load_model_with_patch(model_path)
                return patched
            except Exception as e2:
                st.error(
                    "Tried to patch-load the model (convert 'batch_shape' → 'batch_input_shape') "
                    f"but failed.\n\nPatch loader error: {e2}"
                )
                st.stop()
        # Otherwise show the original error
        st.error(
            "Model load failed. Ensure your requirements pin "
            "tensorflow==2.12.1 and numpy==1.24.3 for legacy H5 models.\n\n"
            f"Loader error: {e}"
        )
        st.stop()

model = load_model()

# ─── Image Preprocessing ────────────────────────────────────────────────────────
def preprocess_image(pil: Image.Image) -> np.ndarray:
    pil = pil.resize((INPUT_SIZE, INPUT_SIZE)).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32)
    arr = preprocess_input(arr)  # InceptionV3 scaling [-1, 1]
    return arr

# ─── Prediction ─────────────────────────────────────────────────────────────────
def classify(pil: Image.Image):
    arr = preprocess_image(pil)
    preds = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(preds))
    return STAGE_LABELS[idx], float(preds[idx])

# ─── UI ─────────────────────────────────────────────────────────────────────────
st.title("Live Drosophila Detection")
st.subheader("📹 Live Camera Detection with Stable Prediction")

if "stable_prediction" not in st.session_state:
    st.session_state["stable_prediction"] = "Waiting..."

# ─── Video Processor ────────────────────────────────────────────────────────────
class StableProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_label = None
        self.count = 0
        self.stable_label = None
        try:
            self.font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        except Exception:
            self.font = ImageFont.load_default()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)

        label, conf = classify(pil)

        # Stability check
        if label == self.last_label:
            self.count += 1
        else:
            self.last_label = label
            self.count = 1

        if self.count >= 3:
            self.stable_label = label
            st.session_state["stable_prediction"] = self.stable_label

        draw = ImageDraw.Draw(pil)
        text = f"{label} ({conf:.0%})"

        try:
            bbox = draw.textbbox((0, 0), text, font=self.font)
        except Exception:
            w, h = draw.textsize(text, font=self.font)
            bbox = (0, 0, w, h)

        padding = 6
        bg_rect = [
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding
        ]
        draw.rectangle(bg_rect, fill="black")
        draw.text((0, 0), text, font=self.font, fill="red")

        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

# ─── Start Webcam ───────────────────────────────────────────────────────────────
webrtc_ctx = webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=StableProcessor,
    async_processing=True
)

# ─── Display Stable Result ──────────────────────────────────────────────────────
st.markdown("### 🧠 Stable Prediction (after 3 consistent frames):")
st.success(st.session_state.get("stable_prediction", "Waiting..."))
