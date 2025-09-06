# =========================
# Drosophila Stage — Live Classifier (Streamlit Cloud hardened)
# =========================

# ---- MUST be set before any TF import or anything that might import Keras ----
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # Force tf.keras (legacy) instead of external keras 3
os.environ["TF_KERAS"] = "1"              # Extra guard for legacy tf.keras
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Avoid GPU lookups on Streamlit Cloud
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Quieter TF logs

# ---- Standard libs ----
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import av
import cv2
import numpy as np
import streamlit as st
from huggingface_hub import HfApi, hf_hub_download
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer


# =========================
# App Config
# =========================
st.set_page_config(page_title="Drosophila Stage — Live Classifier", layout="wide")
st.title("🪰 Drosophila Stage — Live Video Classifier")

HF_REPO = "RishiPTrial/stage_modelv2"
HF_BRANCH = "main"
MODEL_EXTS = (".h5", ".keras")  # allow both, we handle .h5 safest

RTC_CFG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


# =========================
# HF helpers
# =========================
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
            if isinstance(data, dict) and "id_to_label" in data:
                items = sorted(data["id_to_label"].items(), key=lambda kv: int(kv[0]))
                return [str(v) for _, v in items]
        elif path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        pass
    return None


# =========================
# Preprocessing (no keras imports)
# =========================
def infer_arch_from_name(name: str) -> str:
    n = name.lower()
    if "inception" in n:
        return "inceptionv3"
    if "resnet" in n:
        return "resnet50"
    return "generic"

def preprocess_resnet50(rgb_float_0_255: np.ndarray) -> np.ndarray:
    # Keras 'caffe' style: RGB->BGR and mean subtraction (expects 0..255)
    x = rgb_float_0_255[..., ::-1].copy()  # BGR
    x[..., 0] -= 103.939  # B
    x[..., 1] -= 116.779  # G
    x[..., 2] -= 123.68   # R
    return x

def preprocess_inceptionv3(rgb_float_0_255: np.ndarray) -> np.ndarray:
    # Scale to [-1, 1]
    return (rgb_float_0_255 / 127.5) - 1.0

def get_preprocess(model_filename: str) -> Callable[[np.ndarray], np.ndarray]:
    arch = infer_arch_from_name(model_filename)
    if arch == "resnet50":
        return preprocess_resnet50
    if arch == "inceptionv3":
        return preprocess_inceptionv3
    return lambda x: x / 255.0


# =========================
# Model bundle (no tf in annotations)
# =========================
@dataclass
class ModelBundle:
    model: Any
    input_hw: Tuple[int, int]                 # (H, W)
    preprocess: Callable[[np.ndarray], np.ndarray]
    labels: List[str]

# Do NOT import tensorflow at top-level. Import inside the function.
@st.cache_resource(show_spinner=True)
def load_model_bundle(model_path: str, model_filename: str, labels_path: Optional[str]) -> ModelBundle:
    import sys
    # Defensive: if external keras v3 sneaked in, abort cleanly before recursion
    if "keras" in sys.modules:
        kmod = sys.modules["keras"]
        try:
            ver = getattr(kmod, "__version__", "unknown")
        except Exception:
            ver = "unknown"
        st.error(
            f"Detected external `keras` ({ver}) in the environment, which conflicts with `tf.keras`.\n"
            "Please remove `keras` from requirements and pin `tensorflow==2.15.1`.\n"
            "Then clear cache & redeploy."
        )
        st.stop()

    import tensorflow as tf  # safe now; env guards are set above

    # Fail-fast if TF version is not 2.15.x
    if not tf.__version__.startswith("2.15"):
        st.error(f"TensorFlow {tf.__version__} detected. Please pin tensorflow==2.15.1 in requirements.")
        st.stop()

    # Load legacy H5 / Keras model via tf.keras
    model = tf.keras.models.load_model(model_path, compile=False)

    # Input shape
    try:
        ishape = model.input_shape  # (None, H, W, C)
        h = int(ishape[1]) if ishape[1] is not None else None
        w = int(ishape[2]) if ishape[2] is not None else None
    except Exception:
        h = w = None
    if h is None or w is None:
        arch = infer_arch_from_name(model_filename)
        h = w = 299 if arch == "inceptionv3" else 224

    # Number of classes
    try:
        n_classes = int(model.output_shape[-1])
    except Exception:
        n_classes = 2

    # Labels
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

    preprocess = get_preprocess(model_filename)

    return ModelBundle(model=model, input_hw=(h, w), preprocess=preprocess, labels=labels)


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
    cv2.rectangle(frame_bgr, (x - pad, y - th - pad), (x + tw + pad, y + pad), (0, 0, 0), -1)
    cv2.putText(frame_bgr, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame_bgr


# =========================
# Sidebar UI
# =========================
with st.sidebar:
    st.header("⚙️ Settings")
    available_models = list_models(HF_REPO, HF_BRANCH)
    if not available_models:
        st.error("No .h5/.keras models found in your Hugging Face repo.")
        st.stop()

    default_idx = available_models.index("CLEANED_drosophila_stage_resnet50.h5") if "CLEANED_drosophila_stage_resnet50.h5" in available_models else 0
    selected_model = st.selectbox("Select a model", options=available_models, index=default_idx)

    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    infer_every_n = st.slider("Run inference every N frames", 1, 6, 3, 1)
    flip_horizontal = st.checkbox("Mirror webcam (selfie view)", True)
    show_topk = st.slider("Show Top-K classes", 1, 5, 3, 1)

st.caption(f"Models source: {HF_REPO}")


# =========================
# Download + load
# =========================
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

# Warm-up for faster first inference
try:
    dummy = np.zeros((1, bundle.input_hw[0], bundle.input_hw[1], 3), dtype=np.float32)
    _ = bundle.model.predict(dummy, verbose=0)
except Exception:
    pass

st.success(f"Loaded **{selected_model}** | Input: {bundle.input_hw[0]}×{bundle.input_hw[1]} | Classes: {len(bundle.labels)}")


# =========================
# Live video
# =========================
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
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, self.bundle.input_hw[::-1])  # (W, H)
        arr = rgb.astype(np.float32)
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
