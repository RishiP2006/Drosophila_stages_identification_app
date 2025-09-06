import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import av
import cv2
import numpy as np
import streamlit as st
from huggingface_hub import HfApi, hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Use tf.keras only (do NOT pip-install standalone 'keras' to avoid version conflicts on Streamlit Cloud)
import tensorflow as tf

# ----------------------------
# Basic App Config
# ----------------------------
st.set_page_config(page_title="Drosophila Stage Live Classifier", layout="wide")
st.title("🪰 Drosophila Stage — Live Video Classifier")

HF_REPO = "RishiPTrial/stage_modelv2"
HF_BRANCH = "main"
MODEL_EXTS = (".h5",)

# WebRTC STUN (required for Streamlit Cloud / browsers)
RTC_CFG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def list_models(repo_id: str, revision: str) -> List[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, revision=revision)
    return [f for f in files if f.lower().endswith(MODEL_EXTS)]

@st.cache_resource(show_spinner=True)
def download_model(repo_id: str, filename: str, revision: str) -> str:
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)

def _infer_arch_from_name(name: str) -> str:
    n = name.lower()
    if "inception" in n:
        return "inceptionv3"
    if "resnet" in n:
        return "resnet50"
    return "generic"

def _get_preprocess(name: str) -> Callable[[np.ndarray], np.ndarray]:
    arch = _infer_arch_from_name(name)
    if arch == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input as pp
        return pp
    if arch == "inceptionv3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input as pp
        return pp
    # Fallback: scale to [0,1]
    return lambda x: x / 255.0

@dataclass
class ModelBundle:
    model: tf.keras.Model
    input_hw: Tuple[int, int]
    preprocess: Callable[[np.ndarray], np.ndarray]
    labels: List[str]

@st.cache_resource(show_spinner=True)
def load_model_bundle(model_path: str, model_filename: str) -> ModelBundle:
    # Load model (no optimizer/metrics required)
    model = tf.keras.models.load_model(model_path, compile=False)

    # Determine input size
    # Try model.input_shape like (None, H, W, 3)
    try:
        ishape = model.input_shape
        h = int(ishape[1]) if ishape[1] is not None else None
        w = int(ishape[2]) if ishape[2] is not None else None
    except Exception:
        h = w = None

    if h is None or w is None:
        # Reasonable defaults by architecture
        arch = _infer_arch_from_name(model_filename)
        if arch == "inceptionv3":
            h = w = 299
        else:
            h = w = 224

    # Determine number of classes and create default labels if none provided
    try:
        n_classes = int(model.output_shape[-1])
    except Exception:
        # last-resort guess
        n_classes = 2
    labels = [f"Class {i}" for i in range(n_classes)]

    preprocess = _get_preprocess(model_filename)

    return ModelBundle(
        model=model,
        input_hw=(h, w),
        preprocess=preprocess,
        labels=labels
    )

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def draw_label_box(
    frame_bgr: np.ndarray,
    text: str,
    score: float,
    pos: Tuple[int, int] = (10, 30)
) -> np.ndarray:
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

# ----------------------------
# Sidebar Controls
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    available_models = list_models(HF_REPO, HF_BRANCH)
    if not available_models:
        st.error("No .h5 models found in your Hugging Face repo.")
        st.stop()

    selected_model = st.selectbox(
        "Select a model file",
        options=available_models,
        index=available_models.index("CLEANED_drosophila_stage_resnet50.h5")
        if "CLEANED_drosophila_stage_resnet50.h5" in available_models else 0
    )

    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    infer_every_n = st.slider("Run inference every N frames", 1, 6, 3, 1)
    flip_horizontal = st.checkbox("Mirror webcam (selfie view)", True)
    show_topk = st.slider("Show Top-K classes", 1, 5, 3, 1)

st.caption("Repo: " + HF_REPO)

# ----------------------------
# Load/Cache Model
# ----------------------------
with st.spinner(f"Downloading & loading: {selected_model}"):
    local_path = download_model(HF_REPO, selected_model, HF_BRANCH)
    bundle = load_model_bundle(local_path, selected_model)

# Warm-up (build graph for faster first prediction)
# Use tiny dummy input in correct size
try:
    dummy = np.zeros((1, bundle.input_hw[0], bundle.input_hw[1], 3), dtype=np.float32)
    _ = bundle.model.predict(dummy, verbose=0)
except Exception:
    pass

st.success(f"Loaded **{selected_model}** | input: {bundle.input_hw[0]}×{bundle.input_hw[1]} | classes: {len(bundle.labels)}")

# ----------------------------
# WebRTC Processor
# ----------------------------
class LiveVideoProcessor(VideoProcessorBase):
    def __init__(self, bundle: ModelBundle, conf_thr: float, n_skip: int, flip: bool, topk: int):
        self.bundle = bundle
        self.conf_thr = conf_thr
        self.n_skip = max(1, n_skip)
        self.flip = flip
        self.topk = topk
        self._frame_count = 0
        self._last_pred = None  # (label, score, topk list)

    def _predict(self, bgr: np.ndarray):
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.bundle.input_hw[::-1])  # (W,H) order for OpenCV
        arr = img.astype(np.float32)
        arr = self.bundle.preprocess(arr)
        arr = np.expand_dims(arr, axis=0)

        preds = self.bundle.model.predict(arr, verbose=0)
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        preds = np.array(preds).reshape(-1)

        # If model did not include softmax, apply for stable top-k display
        if np.max(preds) > 1.0 or np.min(preds) < 0.0:
            probs = softmax(preds)
        else:
            # assume already probabilities
            probs = preds

        top_indices = np.argsort(probs)[::-1][: self.topk]
        top = [(self.bundle.labels[i] if i < len(self.bundle.labels) else f"Class {i}", float(probs[i])) for i in top_indices]
        label, score = top[0]
        return label, score, top

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if self.flip:
            img = cv2.flip(img, 1)

        self._frame_count += 1
        update_now = (self._frame_count % self.n_skip == 0) or (self._last_pred is None)

        if update_now:
            try:
                self._last_pred = self._predict(img)
            except Exception as e:
                # Draw error once and keep going
                err = f"Inference error: {str(e)[:60]}"
                cv2.putText(img, err, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        if self._last_pred:
            label, score, top = self._last_pred
            if score >= self.conf_thr:
                img = draw_label_box(img, label, score, (10, 30))
            # draw top-k sidebar bar
            y0 = 60
            for i, (lbl, sc) in enumerate(top):
                text = f"{i+1}. {lbl} — {sc:.2f}"
                cv2.putText(img, text, (10, y0 + i*24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ----------------------------
# Start Webcam
# ----------------------------
st.subheader("🎥 Live Webcam")
st.info("Click **Start** to allow webcam. For performance on CPU, the app predicts every Nth frame (tweak in the sidebar).")

webrtc_ctx = webrtc_streamer(
    key=f"webrtc-{selected_model}",  # key includes model name so switching reloads processor
    mode="SENDRECV",
    rtc_configuration=RTC_CFG,
    video_processor_factory=lambda: LiveVideoProcessor(
        bundle=bundle,
        conf_thr=conf_threshold,
        n_skip=infer_every_n,
        flip=flip_horizontal,
        topk=show_topk,
    ),
    media_stream_constraints={"video": True, "audio": False},
)

st.caption(
    "Tip: If you see a black or frozen preview, refresh the page or toggle camera permissions. "
    "Safari/iOS users: make sure you're on HTTPS (Streamlit Cloud is HTTPS by default)."
)
