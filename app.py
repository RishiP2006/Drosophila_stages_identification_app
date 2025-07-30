import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download, HfApi
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import re

st.set_page_config(page_title="Drosophila Gender Detection", layout="centered")
st.title("🩰 Drosophila Gender Detection")
st.write("Upload an image or use live camera. Predictions are made using ensemble of all available `.h5` models.")

HF_REPO_ID = "RishiPTrial/stage_modelv2"

@st.cache_data(show_spinner=False)
def list_h5_models():
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=HF_REPO_ID)
        return [f for f in files if f.lower().endswith(".h5")]
    except Exception as e:
        st.error(f"Failed to list models: {e}")
        return []

@st.cache_resource(show_spinner=False)
def load_all_h5_models():
    try:
        import tensorflow as tf
    except ImportError:
        st.error("TensorFlow not available.")
        return []

    model_paths = [hf_hub_download(repo_id=HF_REPO_ID, filename=name) for name in list_h5_models()]
    models = []
    for path in model_paths:
        try:
            models.append(tf.keras.models.load_model(path))
        except Exception as e:
            st.warning(f"Failed to load {path}: {e}")
    return models

models = load_all_h5_models()
if not models:
    st.error("No `.h5` models found or loaded.")

def preprocess_image_pil(pil_img: Image.Image, size: int = 224):
    arr = pil_img.resize((size, size))
    arr = np.asarray(arr).astype(np.float32) / 255.0
    return arr

def ensemble_predict(models, img_array):
    x = np.expand_dims(img_array, axis=0)
    predictions = [model.predict(x)[0][0] for model in models]
    avg_pred = np.mean(predictions)
    label = "Female" if avg_pred >= 0.5 else "Male"
    confidence = avg_pred if label == "Female" else 1 - avg_pred
    return label, confidence

# Upload Image Section
st.markdown("---")
st.subheader("📷 Upload Image")
img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if img_file and models:
    pil_img = Image.open(img_file).convert("RGB")
    st.image(pil_img, use_column_width=True)
    arr = preprocess_image_pil(pil_img)
    label, prob = ensemble_predict(models, arr)
    st.success(f"Prediction: {label} ({prob:.1%})")

# Live Camera Section
st.markdown("---")
st.subheader("📸 Live Camera Gender Detection")

class GenderDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.models = models

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        except Exception:
            font = ImageFont.load_default()

        arr = preprocess_image_pil(pil)
        label, prob = ensemble_predict(self.models, arr)
        draw.text((10, 10), f"{label} ({prob:.1%})", fill="red", font=font)

        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

if models:
    webrtc_streamer(
        key="live-gender-detect",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=GenderDetectionProcessor,
        async_processing=True,
    )
else:
    st.warning("Please upload valid `.h5` models to Hugging Face first.")

st.markdown("---")
st.write("**Notes:**")
st.write(f"- Models are loaded from Hugging Face repo: `{HF_REPO_ID}`")
st.write("- Ensemble of all available `.h5` models is used.")
st.write("- Streamlit WebRTC + PIL is used for real-time processing.")
