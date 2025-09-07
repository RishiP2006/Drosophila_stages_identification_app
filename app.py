# =========================
# Drosophila Stage — Live Video Classifier (Keras 3, robust + minimal)
# =========================

# --- Environment BEFORE any Keras/TF import ---
import os
os.environ["KERAS_BACKEND"] = "tensorflow"   # Keras 3 uses TensorFlow backend
os.environ["CUDA_VISIBLE_DEVICES"] = ""       # no GPU probing on Streamlit Cloud
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # quieter TF logs

# --- Standard libs ---
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
MODEL_EXTS = (".keras",)  # we only use Keras 3 native models

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
# Preprocessing helpers (inline; no keras.applications imports at runtime)
# =========================
def _arch_from_name(name: str) -> str:
    n = name.lower()
    if "convnext" in n: return "convnext"
    if "resnet"   in n: return "resnet50"
    if "inception" in n: return "inceptionv3"
    return "generic"

def preprocess_resnet50(rgb_0_255: np.ndarray) -> np.ndarray:
    # "caffe" style: RGB->BGR + mean subtraction (expects 0..255 input)
    x = rgb_0_255[..., ::-1].copy()  # to BGR
    x[..., 0] -= 103.939  # B
    x[..., 1] -= 116.779  # G
    x[..., 2] -= 123.68   # R
    return x

def preprocess_convnext(rgb_0_255: np.ndarray) -> np.ndarray:
    # Match your training (ImageDataGenerator with rescale=1./255)
    return rgb_0_255 / 255.0

def preprocess_inceptionv3(rgb_0_255: np.ndarray) -> np.ndarray:
    return (rgb_0_255 / 127.5) - 1.0

def get_default_preprocess_for_filename(model_filename: str) -> Callable[[np.ndarray], np.ndarray]:
    a = _arch_from_name(model_filename)
    if a == "resnet50":    return preprocess_resnet50
    if a == "convnext":    return preprocess_convnext
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
    input_hw: Tuple[int, int]  # (H, W)
    preprocess: Callable[[np.ndarray], np.ndarray]
    labels: List[str]

def _has_input_preprocessing_layers(model: Any) -> bool:
    """
    Detect if the graph already contains input preprocessing (Lambda/Rescaling/Normalization)
    near the input. If yes, we shouldn't preprocess again in the app.
    """
    try:
        first_layers = model.layers[:6]  # look at a few early layers
        for lyr in first_layers:
            t = lyr.__class__.__name__
            if t in ("Lambda", "Rescaling", "Normalization"):
                return True
    except Exception:
        pass
    return False

@st.cache_resource(show_spinner=True)
def load_model_bundle(model_path: str, model_filename: str) -> ModelBundle:
    import keras
    ver = getattr(keras, "__version__", "unknown")
    if not ver.startswith("3."):
        st.error(f"Detected keras=={ver}. Pin keras==3.3.2 (or compatible 3.x) and TF>=2.16.")
        st.stop()

    # 1) Try normal safe load
    try:
        model = keras.saving.load_model(model_path, compile=False)  # safe by default
    except Exception:
        # 2) Retry with unsafe deserialization + custom object for 'preprocess_input'
        try:
            # Enable unsafe mode (needed for Lambda / Python callables)
            try:
                from keras import config as KCFG
                KCFG.enable_unsafe_deserialization()
            except Exception:
                pass  # older 3.x will still honor safe_mode=False below

            custom_objects = {
                # Many TF/Keras 2.x models used Lambda(preprocess_input). The serialized name is 'preprocess_input'.
                "preprocess_input": preprocess_resnet50,
            }
            model = keras.saving.load_model(
                model_path,
                compile=False,
                custom_objects=custom_objects,
                safe_mode=False,
            )
        except Exception as e2:
            st.error(f"Failed to load model '{model_filename}'. Error:\n{e2}")
            st.stop()

    # Infer input size
    try:
        ishape = model.inputs[0].shape  # (None, H, W, C)
        h = int(ishape[1]); w = int(ishape[2])
    except Exception:
        a = _arch_from_name(model_filename)
        h = w = 224 if a in ("resnet50", "convnext", "generic") else 299

    # Classes & default labels
    try:
        oshape = model.outputs[0].shape
        n_classes = int(oshape[-1])
    except Exception:
        n_classes = 2
    labels = [f"Class {i}" for i in range(n_classes)]

    # Decide external preprocessing:
    # If the graph already has Lambda/Rescaling/Normalization near the input, use identity.
    if _has_input_preprocessing_layers(model):
        preprocess = lambda x: x  # model handles it internally (e.g., your ResNet50 with Lambda)
    else:
        preprocess = get_default_preprocess_for_filename(model_filename)

    return ModelBundle(model=model, input_hw=(h, w), preprocess=preprocess, labels=labels)


# =========================
# Sidebar — minimal: pick model
# =========================
with st.sidebar:
    st.header("Model")
    models = list_models(HF_REPO, HF_BRANCH)
    if not models:
        st.error("No .keras models found in your Hugging Face repo.")
        st.stop()

    # Prefer your two models explicitly if present
    preferred_order = [
        "drosophila_stage_resnet50_finetuned.keras",
        "best_convnext_model.keras",
    ]
    default_idx = next((models.index(m) for m in preferred_order if m in models), 0)
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
# Live video — predict every frame (top-1 only)
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
        arr = self.bundle.preprocess(arr)  # either identity or proper normalization
        arr = np.expand_dims(arr, axis=0)

        preds = self.bundle.model.predict(arr, verbose=0)
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        preds = np.array(preds).reshape(-1)

        # if logits, softmax; if already probs, numerically identical
        probs = softmax_safe(preds) if (np.max(preds) > 1.0 or np.min(preds) < 0.0) else preds
        i = int(np.argmax(probs))
        return self.bundle.labels[i], float(probs[i])

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # selfie view
        try:
            label, score = self._predict_top1(img)
            img = draw_label_box(img, label, score, (10, 30))
        except Exception as e:
            cv2.putText(img, f"Inference error: {str(e)[:60]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.subheader("🎥 Live Webcam")
webrtc_streamer(
    key=f"webrtc-{selected_model}",
    mode="SENDRECV",
    rtc_configuration=RTC_CFG,
    video_processor_factory=lambda: LiveVideoProcessor(bundle=bundle),
    media_stream_constraints={"video": True, "audio": False},
)

st.caption("If preview is black/frozen, refresh or toggle camera permissions. HTTPS is required for webcam.")
