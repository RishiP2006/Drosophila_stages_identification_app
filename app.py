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

HF_REPO_ID = "RishiPTrial/stage_modelv2"

@st.cache_data(show_spinner=False)
def list_hf_models():
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=HF_REPO_ID)
        return [f for f in files if f.lower().endswith(".h5") and not f.startswith(".")]
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def build_models_info():
    files = list_hf_models()
    info = {}
    for fname in files:
        input_size = 224
        if "inceptionv3" in fname.lower():
            input_size = 299
        info[fname] = {"type": "classification", "framework": "keras", "input_size": input_size}
    return info

MODELS_INFO = build_models_info()
if not MODELS_INFO:
    st.error(f"No model files found in HF repo {HF_REPO_ID}")

@st.cache_resource(show_spinner=False)
def load_model_from_hf(name, info):
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=name)
    except Exception as e:
        st.error(f"Error downloading {name}: {e}")
        return None

    try:
        import tensorflow as tf
        from tensorflow.keras.applications.inception_v3 import preprocess_input as iv3
        from tensorflow.keras.applications.convnext import preprocess_input as cx
        from tensorflow.keras.applications.resnet50 import preprocess_input as res

        custom_objects = {
            "preprocess_input": iv3,
            "iv3_preprocess": iv3,
            "cx_preprocess": cx,
            "resnet_preprocess": res,
        }
        model = tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Failed loading Keras model {name}: {e}")
        return None

def preprocess_image_pil(pil_img: Image.Image, size: int):
    arr = pil_img.resize((size, size))
    arr = np.asarray(arr).astype(np.float32) / 255.0
    return arr

def classify(model, img_array: np.ndarray):
    x = np.expand_dims(img_array, axis=0)
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            return model.predict(x)
    except Exception:
        pass
    st.error("Unknown model type for prediction.")
    return None

STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa", "black pupa"
]

def interpret_stage(preds):
    if preds is None:
        return None, None
    arr = np.asarray(preds)
    if arr.ndim == 2 and arr.shape[1] == len(STAGE_LABELS):
        idx = int(np.argmax(arr))
        return STAGE_LABELS[idx], float(arr[0][idx])
    st.warning(f"Unexpected prediction shape: {arr.shape}")
    return None, None

class StageDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.info = MODELS_INFO[model_name]

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        if self.model is not None:
            size = self.info.get("input_size", 224)
            arr = preprocess_image_pil(pil, size)
            preds = classify(self.model, arr)
            label, prob = interpret_stage(preds)
            if label:
                draw.text((10, 10), f"{label} ({prob:.1%})", fill="red")
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

def safe_label(name):
    return re.sub(r"[^\w\s.-]", "_", name)

safe_to_real = {safe_label(n): n for n in MODELS_INFO}
choice = st.selectbox("Select model", list(safe_to_real.keys())) if MODELS_INFO else None
model_name = safe_to_real.get(choice)
model = None
if model_name:
    info = MODELS_INFO[model_name]
    model = load_model_from_hf(model_name, info)

st.markdown("---")
st.subheader("📷 Upload Image")
img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if img_file and model is not None:
    pil_img = Image.open(img_file).convert("RGB")
    st.image(pil_img, use_column_width=True)
    info = MODELS_INFO[model_name]
    arr = preprocess_image_pil(pil_img, info.get("input_size", 224))
    preds = classify(model, arr)
    label, prob = interpret_stage(preds)
    if label:
        st.success(f"Prediction: {label} ({prob:.1%})")

st.markdown("---")
st.subheader("📸 Live Camera Stage Detection")
if model is not None:
    ctx = webrtc_streamer(
        key="live-stage-detect",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=StageDetectionProcessor,
        async_processing=True,
    )
else:
    st.warning("Please select a model first.")

st.markdown("---")
st.write("**Notes:**")
st.write(f"- Models from HF: {HF_REPO_ID}")
st.write("- Only classification models (.h5) are supported.")
st.write("- Live camera uses PIL for drawing, no font styling for speed.")
