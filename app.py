# =========================
# Drosophila Stage — Live Video Classifier (Streamlit Cloud ready)
# =========================

# --- MUST set env vars BEFORE any TensorFlow/Keras import happens ---
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # use legacy tf.keras (for .h5 saved with TF 2.x)
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # avoid GPU lookups on Streamlit Cloud

import time
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Any

import av
import cv2
import numpy as np
import streamlit as st
from huggingface_hub import HfApi, hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


# ----------------------------
# Streamlit App Config
# ----------------------------
st.set_page_config(page_title="Drosophila Stage — Live Classifier", layout="wide")
st.title("🪰 Drosophila Stage — Live Video Classifier")

HF_REPO = "RishiPTrial/stage_modelv2"
HF_BRANCH = "main"
MODEL_EXTS = (".h5",)

RTC_CFG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})


# ----------------------------
# HF helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def list_models(repo_id: str, revision: str) -> List[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, revision=revision)
    return [f for f in files if f.lower().endswith(MODEL_EXTS)]

@st.cache_data(show_spinner=False)
def list_label_files(repo_id: str, revision: str) -> List[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, revision=revision)
    return [f for f in files if f.endswith(("labels.json", "classes.txt"))]

@st.cache_resource(show_spinner=True)
def download_from_hf(repo_id: str, filename: str, revision: str) -> str:
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)

def try_load_labels(path: str) -> Optional[List[str]]:
    try:
        if path.endswith(".json"):
            import json
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(x) for x in data]
            if isinstance(data, dict) and "id_to_label" in data and isinstance(data["id_to_label"], dict):
                items = sorted(data["id_to_label"].items(), key=lambda kv: int(kv[0]))
                return [str(v) for _, v in items]
        elif path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        pass
    return None


# ----------------------------
# Model / preprocess helpers
# ----------------------------
def _infer_arch_from_name(name: str) -> str:
    n = name.lower()
    if "inception" in n:
        return "inceptionv3"
    if "resnet" in n:
        return "resnet50"
    return "generic"

def _get_preprocess(name: str) -> Callable[[np.ndarray], np.ndarray]:
    # Import inside the function to avoid touching TF/Keras until actually needed
    arch = _infer_arch_from_name(name)
    if arch == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input as pp
        return pp
    if arch == "inceptionv3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input as pp
        return pp
    return lambda x: x / 255.0

@dataclass
class ModelBundle:
    # Avoid tf.keras in annotations to prevent lazy-loader recursion
    model: Any
    input_hw: Tuple[int, int]                               # (H, W)
    preprocess: Callable[[np.ndarray], np.ndarray]
    labels: List[str]

@st.cache_resource(show_spinner=True)
def load_model_bundle(model_path: str, model_filename: str, labels_path: Optional[str]) -> ModelBundle:
    # Import TensorFlow here (after env vars) to keep things safe on Cloud
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)

    # Input size
    try:
        ishape = model.input_shape  # (None, H, W, C)
        h = int(ishape[1]) if ishape[1] is not None else None
        w = int(ishape[2]) if ishape[2] is not None else None
    except Exception:
        h = w = None

    if h is None or w is None:
        arch = _infer_arch_from_name(model_filename)
        h = w = 299 if arch == "inceptionv3" else 224

    # Classes
    try:
        n_classes = int(model.output_shape[-1])
    except Exception:
        n_classes = 2

    labels: Optional[List[str]] = None
    if labels_path:
        labels = try_load_labels(labels_path)
    if not labels:
        labels = [f"Class {i}" for i in range(n_classes)]
    else:
        if len(labels) < n_classes:
            labels = labels + [f"Class {i}" for i in range(len(labels), n_classes)]
        elif len(labels) > n_classes:
            labels = labels[:n_classes]

    preprocess = _get_preprocess(model_filename)

    return ModelBundle(
        model=model,
        input_hw=(h, w),
        preprocess=preprocess,
        labels=labels
    )

def softmax_safe(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    return e / s if s != 0 else np.zeros_like(x)

def draw_label_box(frame_bgr: np.ndarray, text: str, score: float, pos=(10, 30)) -> np.ndarray:
    x, y = pos
    label = f"{text}: {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    pad = 6
    cv2.rectangle(frame_bgr, (x - pad, y - th - pad), (x + tw + pad), (y + pad), (0, 0, 0), -1)
    cv2.putText(frame_bgr, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame_bgr


# ----------------------------
# Sidebar UI
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    available_models = list_models(HF_REPO, HF_BRANCH)
    if not available_models:
        st.error("No .h5 models found in your Hugging Face repo.")
        st.stop()

    default_idx = available_models.index("CLEANED_drosophila_stage_resnet50.h5") if "CLEANED_drosophila_stage_resnet50.h5" in available_models else 0
    selected_model = st.selectbox("Select a model file", options=available_models, index=default_idx)

    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    infer_every_n = st.slider("Run inference every N frames", 1, 6, 3, 1)
    flip_horizontal = st.checkbox("Mirror webcam (selfie view)", True)
    show_topk = st.slider("Show Top-K classes", 1, 5, 3, 1)

st.caption(f"Models source: {HF_REPO}")


# ----------------------------
# Download + load model (& optional labels)
# ----------------------------
with st.spinner(f"Downloading & loading: {selected_model}"):
    model_local_path = download_from_hf(HF_REPO, selected_model, HF_BRANCH)

    labels_local_path = None
    lbl_files = list_label_files(HF_REPO, HF_BRANCH)
    if lbl_files:
        preferred = None
        for cand in ("labels.json", "classes.txt"):
            for f in lbl_files:
                if f.endswith(cand):
                    preferred = f
                    break
            if preferred:
                break
        if preferred:
            labels_local_path = download_from_hf(HF_REPO, preferred, HF_BRANCH)

    bundle = load_model_bundle(model_local_path, selected_model, labels_local_path)

# Warm-up (faster first inference)
try:
    dummy = np.zeros((1, bundle.input_hw[0], bundle.input_hw[1], 3), dtype=np.float32)
    _ = bundle.model.predict(dummy, verbose=0)
except Exception:
    pass

st.success(f"Loaded **{selected_model}** | Input: {bundle.input_hw[0]}×{bundle.input_hw[1]} | Classes: {len(bundle.labels)}")


# ----------------------------
# Live video processor
# ----------------------------
class LiveVideoProcessor(VideoProcessorBase):
    def __init__(self, bundle: ModelBundle, conf_thr: float, n_skip: int, flip: bool, topk: int):
        self.bundle = bundle
        self.conf_thr = conf_thr
        self.n_skip = max(1, n_skip)
        self.flip = flip
        self.topk = topk
        self._frame_count = 0
        self._last_pred = None  # (label, score, top list)

    def _predict(self, bgr: np.ndarray):
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.bundle.input_hw[::-1])  # cv2 wants (W, H)
        arr = img.astype(np.float32)
        arr = self.bundle.preprocess(arr)
        arr = np.expand_dims(arr, axis=0)

        preds = self.bundle.model.predict(arr, verbose=0)
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        preds = np.array(preds).reshape(-1)

        probs = softmax_safe(preds) if (np.max(preds) > 1.0 or np.min(preds) < 0.0) else preds
        top_indices = np.argsort(probs)[::-1][: self.topk]
        top = [(self.bundle.labels[i] if i < len(self.bundle.labels) else f"Class {i}", float(probs[i])) for i in top_indices]
        label, score = top[0]
        return label, score, top

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if self.flip:
            img = cv2.flip(img, 1)

        self._frame_count += 1
        need_update = (self._frame_count % self.n_skip == 0) or (self._last_pred is None)

        if need_update:
            try:
                self._last_pred = self._predict(img)
            except Exception as e:
                cv2.putText(img, f"Inference error: {str(e)[:60]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        if self._last_pred:
            label, score, top = self._last_pred
            if score >= self.conf_thr:
                img = draw_label_box(img, label, score, (10, 30))
            y0 = 60
            for i, (lbl, sc) in enumerate(top):
                cv2.putText(img, f"{i+1}. {lbl} — {sc:.2f}", (10, y0 + i*24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ----------------------------
# UI: Webcam
# ----------------------------
st.subheader("🎥 Live Webcam")
st.info("Click **Start** to allow webcam. On CPU, the app predicts every Nth frame (tweak in the sidebar).")

webrtc_ctx = webrtc_streamer(
    key=f"webrtc-{selected_model}",
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

st.caption("If preview is black/frozen, refresh or toggle camera permissions. HTTPS is required for webcam.")
