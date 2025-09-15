
import os
os.environ["KERAS_BACKEND"] = "tensorflow"   
os.environ["CUDA_VISIBLE_DEVICES"] = ""     
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   

# --- Standard libs ---
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import av
import cv2
import json
import numpy as np
import streamlit as st
from huggingface_hub import HfApi, hf_hub_download
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
)


st.set_page_config(page_title="Drosophila Stage — Live Classifier", layout="wide")
st.title("🪰 Drosophila Stage — Live Video Classifier")

HF_REPO = "RishiPTrial/stage_modelv2"
HF_BRANCH = "main"
MODEL_EXTS = (".keras",)
LABEL_FILE_CANDIDATES = ["labels.json", "classes.txt", "class_names.txt"]

# Fallback labels for your 8 classes (order matches your training folder names 1..8)
CANONICAL_8_LABELS = [
    "Egg",
    "First Instar",
    "Second Instar",
    "Third Instar",
    "White Pupa",
    "Brown Pupa",
    "Eye Pupa",
    "Black Pupa",
]

RTC_CFG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})



@st.cache_data(show_spinner=False)
def list_models(repo_id: str, revision: str) -> List[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, revision=revision)
    return [f for f in files if f.lower().endswith(MODEL_EXTS)]

@st.cache_resource(show_spinner=True)
def download_from_hf(repo_id: str, filename: str, revision: str) -> str:
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)

@st.cache_data(show_spinner=False)
def try_load_labels(repo_id: str, revision: str) -> Optional[List[str]]:
    """Try to fetch labels from common filenames in the HF repo."""
    for fname in LABEL_FILE_CANDIDATES:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=fname, revision=revision)
        except Exception:
            continue
        try:
            if fname.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    arr = json.load(f)
                if isinstance(arr, list) and all(isinstance(s, str) for s in arr):
                    labels = [s.strip() for s in arr if s.strip()]
                    if labels:
                        return labels
            else:
                with open(path, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f.readlines()]
                labels = [ln for ln in lines if ln]
                if labels:
                    return labels
        except Exception:
            continue
    return None



def _arch_from_name(name: str) -> str:
    n = name.lower()
    if "convnext" in n: return "convnext"
    if "resnet"   in n: return "resnet50"
    if "inception" in n: return "inceptionv3"
    return "generic"

def preprocess_resnet50(rgb_0_255: np.ndarray) -> np.ndarray:
    # caffe-style: RGB->BGR + mean subtraction; expects uint/float 0..255 input
    x = rgb_0_255[..., ::-1].copy()  # to BGR
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x

def preprocess_convnext(rgb_0_255: np.ndarray) -> np.ndarray:
    # matches your training pipeline (ImageDataGenerator rescale=1./255)
    return rgb_0_255 / 255.0

def preprocess_inceptionv3(rgb_0_255: np.ndarray) -> np.ndarray:
    return (rgb_0_255 / 127.5) - 1.0

def default_preprocess_for_filename(model_filename: str) -> Callable[[np.ndarray], np.ndarray]:
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



@dataclass
class ModelBundle:
    model: Any
    input_hw: Tuple[int, int]      # (H, W)
    preprocess: Callable[[np.ndarray], np.ndarray]
    labels: List[str]

def _has_input_preproc_layers(model: Any) -> bool:
    """Detect early Lambda/Rescaling/Normalization to avoid double preprocessing."""
    try:
        for lyr in model.layers[:6]:
            if lyr.__class__.__name__ in ("Lambda", "Rescaling", "Normalization"):
                return True
    except Exception:
        pass
    return False

@st.cache_resource(show_spinner=True)
def load_model_bundle(model_path: str, model_filename: str, repo_id: str, revision: str) -> ModelBundle:
    import keras, tensorflow as tf
    ver = getattr(keras, "__version__", "unknown")
    if not ver.startswith("3."):
        st.error(f"Detected keras=={ver}. Pin keras==3.3.2 and tensorflow==2.16.1.")
        st.stop()

    # 1) Try safe load first
    replaced_lambda = False
    try:
        model = keras.saving.load_model(model_path, compile=False)  # safe by default
    except Exception:
        # 2) Retry: map serialized 'preprocess_input' Lambda to identity to bypass shape inference
        def _identity(x):
            return tf.identity(x)
        try:
            try:
                from keras import config as KCFG
                KCFG.enable_unsafe_deserialization()
            except Exception:
                pass
            model = keras.saving.load_model(
                model_path,
                compile=False,
                custom_objects={"preprocess_input": _identity},
                safe_mode=False,
            )
            replaced_lambda = True
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

    # Determine number of classes
    try:
        oshape = model.outputs[0].shape
        n_classes = int(oshape[-1])
    except Exception:
        n_classes = 2

    # Load labels (HF > fallback)
    labels = try_load_labels(repo_id, revision)
    if labels is None and n_classes == 8:
        labels = CANONICAL_8_LABELS.copy()
    if labels is None or len(labels) != n_classes:
        # final fallback to generic labels if mismatch
        labels = [f"Class {i}" for i in range(n_classes)]

    # Decide external preprocessing:
    if replaced_lambda:
        preprocess = default_preprocess_for_filename(model_filename)
    else:
        preprocess = (lambda x: x) if _has_input_preproc_layers(model) else default_preprocess_for_filename(model_filename)

    return ModelBundle(model=model, input_hw=(h, w), preprocess=preprocess, labels=labels)



with st.sidebar:
    st.header("Model")
    models = list_models(HF_REPO, HF_BRANCH)
    if not models:
        st.error("No .keras models found in your Hugging Face repo.")
        st.stop()

    preferred = [
        "drosophila_stage_resnet50_finetuned.keras",
        "best_convnext_model.keras",
    ]
    default_idx = next((models.index(m) for m in preferred if m in models), 0)
    selected_model = st.selectbox("Select a model", options=models, index=default_idx)

st.caption(f"Models source: {HF_REPO}")

# Download + load
with st.spinner(f"Downloading & loading: {selected_model}"):
    model_local_path = download_from_hf(HF_REPO, selected_model, HF_BRANCH)
    bundle = load_model_bundle(model_local_path, selected_model, HF_REPO, HF_BRANCH)

# Optional warm-up
try:
    dummy = np.zeros((1, bundle.input_hw[0], bundle.input_hw[1], 3), dtype=np.float32)
    _ = bundle.model.predict(dummy, verbose=0)
except Exception:
    pass




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
        arr = self.bundle.preprocess(arr)                   # external preprocessing (or identity)
        arr = np.expand_dims(arr, axis=0)

        preds = self.bundle.model.predict(arr, verbose=0)
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        preds = np.array(preds).reshape(-1)

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
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CFG,
    video_processor_factory=lambda: LiveVideoProcessor(bundle=bundle),
    media_stream_constraints={"video": True, "audio": False},
)

st.caption("If the preview is black/frozen, refresh or toggle camera permissions. HTTPS is required for webcam.")
