# =========================
# Drosophila Stage — Live Classifier (minimal UI, Streamlit Cloud safe)
# =========================

# ---- Env (must be set before any TF/Keras import) ----
import os, sys
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def _get_tf_with_keras_alias():
    """
    Import TensorFlow and alias any future `import keras` to `tf.keras`
    to avoid keras v3 recursion / lazy-loader issues.
    """
    for k in list(sys.modules.keys()):
        if k == "keras" or k.startswith("keras."):
            del sys.modules[k]
    import tensorflow as tf
    sys.modules["keras"] = tf.keras
    return tf

# ---- Regular imports ----
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import av
import cv2
import numpy as np
import streamlit as st
from huggingface_hub import HfApi, hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


# =========================
# App Config
# =========================
st.set_page_config(page_title="Drosophila Stage — Live Classifier", layout="wide")
st.title("🪰 Drosophila Stage — Live Video Classifier")

HF_REPO = "RishiPTrial/stage_modelv2"
HF_BRANCH = "main"
MODEL_EXTS = (".h5",)  # keep to .h5 for tf.keras 2.15 compatibility

RTC_CFG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


# =========================
# Hugging Face helpers
# =========================
@st.cache_data(show_spinner=False)
def list_models(repo_id: str, revision: str) -> List[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, revision=revision)
    return [f for f in files if f.lower().endswith(MODEL_EXTS)]

@st.cache_resource(show_spinner=True)
def download_from_hf(repo_id: str, filename: str, revision: str) -> str:
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


# =========================
# Preprocess (inline; no keras imports)
# =========================
def _arch_from_name(name: str) -> str:
    n = name.lower()
    if "inception" in n: return "inceptionv3"
    if "resnet" in n:    return "resnet50"
    return "generic"

def preprocess_resnet50(rgb_0_255: np.ndarray) -> np.ndarray:
    # RGB->BGR + mean subtraction (caffe style)
    x = rgb_0_255[..., ::-1].copy()
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x

def preprocess_inceptionv3(rgb_0_255: np.ndarray) -> np.ndarray:
    return (rgb_0_255 / 127.5) - 1.0

def get_preprocess(model_filename: str) -> Callable[[np.ndarray], np.ndarray]:
    a = _arch_from_name(model_filename)
    if a == "resnet50":    return preprocess_resnet50
    if a == "inceptionv3": return preprocess_inceptionv3
    return lambda x: x / 255.0

def softmax_safe(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    return e / s if s != 0 else np.zeros_like(x)


# =========================
# Model bundle
# =========================
@dataclass
class ModelBundle:
    model: Any
    input_hw: Tuple[int, int]                 # (H, W)
    preprocess: Callable[[np.ndarray], np.ndarray]
    labels: List[str]

@st.cache_resource(show_spinner=True)
def load_model_bundle(model_path: str, model_filename: str) -> ModelBundle:
    tf = _get_tf_with_keras_alias()
    if not tf.__version__.startswith("2.15"):
        st.error(f"TensorFlow {tf.__version__} detected. Please pin tensorflow==2.15.1.")
        st.stop()

    model = tf.keras.models.load_model(model_path, compile=False)

    # Input size
    try:
        ishape = model.input_shape  # (None, H, W, C)
        h = int(ishape[1]) if ishape[1] is not None else None
        w = int(ishape[2]) if ishape[2] is not None else None
    except Exception:
        h = w = None
    if h is None or w is None:
        a = _arch_from_name(model_filename)
        h = w = 299 if a == "inceptionv3" else 224

    # Classes & default labels
    try:
        n_classes = int(model.output_shape[-1])
    except Exception:
        n_classes = 2
    labels = [f"Class {i}" for i in range(n_classes)]

    preprocess = get_preprocess(model_filename)
    return ModelBundle(model=model, input_hw=(h, w), preprocess=preprocess, labels=labels)


# =========================
# UI — Minimal: just pick model
# =========================
with st.sidebar:
    st.header("Model")
    models = list_models(HF_REPO, HF_BRANCH)
    if not models:
        st.error("No .h5 models found in your Hugging Face repo.")
        st.stop()
    default_idx = models.index("CLEANED_drosophila_stage_resnet50.h5") if "CLEANED_drosophila_stage_resnet50.h5" in models else 0
    selected_model = st.selectbox("Select a model", options=models, index=default_idx)

st.caption(f"Models source: {HF_REPO}")

# Download + load
with st.spinner(f"Downloading & loading: {selected_model}"):
    model_local_path = download_from_hf(HF_REPO, selected_model, HF_BRANCH)
    bundle = load_model_bundle(model_local_path, selected_model)

# Optional warm-up
try:
    dummy = np.zeros((1, bundle.input_hw[0], bundle.input_hw[1], 3), dtype=np.float32)
    _ = bundle.model.predict(dummy, verbose=0)
except Exception:
    pass

st.success(f"Loaded **{selected_model}** | Input: {bundle.input_hw[0]}×{bundle.input_hw[1]} | Classes: {len(bundle.labels)}")


# =========================
# Video processor — per-frame prediction (no extra controls)
# =========================
def draw_label_box(frame_bgr: np.ndarray, text: str, score: float, pos=(10, 30)) -> np.ndarray:
    x, y = pos
    label = f"{text}: {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    pad = 6
    cv2.rectangle(frame_bgr, (x - pad, y - th - pad), (x + tw + pad, y + pad), (0, 0, 0), -1)
    cv2.putText(frame_bgr, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame_bgr

class LiveVideoProcessor(VideoProcessorBase):
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle

    def _predict_top1(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, self.bundle.input_hw[::-1])  # (W, H)
        arr = rgb.astype(np.float32)
        arr = self.bundle.preprocess(arr)
        arr = np.expand_dims(arr, axis=0)

        preds = self.bundle.model.predict(arr, verbose=0)
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        preds = np.array(preds).reshape(-1)

        probs = softmax_safe(preds) if (np.max(preds) > 1.0 or np.min(preds) < 0.0) else preds
        i = int(np.argmax(probs))
        return self.bundle.labels[i], float(probs[i])

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Mirror for natural selfie view
        img = cv2.flip(img, 1)

        try:
            label, score = self._predict_top1(img)
            img = draw_label_box(img, label, score, (10, 30))
        except Exception as e:
            cv2.putText(img, f"Inference error: {str(e)[:60]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =========================
# Webcam
# =========================
st.subheader("🎥 Live Webcam")
webrtc_streamer(
    key=f"webrtc-{selected_model}",
    mode="SENDRECV",
    rtc_configuration=RTC_CFG,
    video_processor_factory=lambda: LiveVideoProcessor(bundle=bundle),
    media_stream_constraints={"video": True, "audio": False},
)

st.caption("Tip: If preview is black/frozen, refresh or toggle camera permissions. HTTPS is required for webcam.")
