# ------------------ app.py ------------------
# Force CPU + quiet TF logs BEFORE any TF/Keras import
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
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
from tensorflow.keras.models import load_model as tf_load_model
from keras.models import load_model as k_load_model

# ─── Config ────────────────────────────────────────────────────────────────────
HF_REPO_ID = "RishiPTrial/my-model-name"
CANDIDATE_FILES = [
    "drosophila_inceptionv3_classifier.keras",   # preferred (upload this if you can)
    "drosophila_inceptionv3_classifier.h5",      # your current file
]
INPUT_SIZE = 299
STAGE_LABELS = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa"
]

# ─── RTC config: use STUN by default; read TURN from secrets if provided ───────
def get_rtc_configuration():
    # Optionally set in .streamlit/secrets.toml:
    # [ice]
    # policy = "all"        # or "relay" to force TURN
    # servers = [
    #   {urls = ["stun:stun.l.google.com:19302", "stun:global.stun.twilio.com:3478"]},
    #   {urls = ["turn:YOUR_TURN:3478?transport=udp","turns:YOUR_TURN:5349"], username="USER", credential="PASS"}
    # ]
    try:
        ice = st.secrets.get("ice", None)
        if ice and "servers" in ice:
            cfg = {"iceServers": ice["servers"]}
            if "policy" in ice:
                cfg["iceTransportPolicy"] = ice["policy"]
            return cfg
    except Exception:
        pass
    # sensible default STUN (works for many networks; add TURN for restrictive ones)
    return {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302", "stun:global.stun.twilio.com:3478"]}
        ]
    }

# ─── Model loading (robust to legacy H5) ───────────────────────────────────────
def _download_first_existing(repo_id, candidates):
    last_err = None
    for fname in candidates:
        try:
            return hf_hub_download(repo_id=repo_id, filename=fname, token=st.secrets.get("HF_TOKEN"))
        except Exception as e:
            last_err = e
    raise last_err or RuntimeError("No model file found in Hugging Face repo.")

def build_head(num_classes, base_weights="imagenet"):
    base = InceptionV3(include_top=False, weights=base_weights, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
    x = layers.GlobalAveragePooling2D()(base.output)
    out = layers.Dense(num_classes, activation="softmax", name="stage_head")(x)
    return Model(base.input, out)

@st.cache_resource(show_spinner="Loading model from Hugging Face…")
def load_or_fallback_model():
    # Try to download any candidate file
    try:
        model_path = _download_first_existing(HF_REPO_ID, CANDIDATE_FILES)
    except Exception as e:
        st.warning(f"Could not download a model file from HF: {e}\nUsing ImageNet-initialized fallback.")
        return build_head(len(STAGE_LABELS), base_weights="imagenet")

    tf_err = None
    # 1) Try tf.keras full-model loader
    try:
        return tf_load_model(model_path, compile=False)
    except Exception as e:
        tf_err = e

    # 2) Try Keras 3 legacy H5 loader
    k_err = None
    try:
        return k_load_model(model_path, compile=False)
    except Exception as e2:
        k_err = e2

    # 3) Build clean graph, then try to load weights by name (partial OK)
    model = build_head(len(STAGE_LABELS), base_weights="imagenet")
    loaded_from_h5 = False
    try:
        # This works if the H5 contains a "model_weights" group (even if it was a full-model save).
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        loaded_from_h5 = True
    except Exception:
        pass

    msg = (
        "Your legacy .h5 model could not be deserialized (multi-input Dense in graph). "
        "The app is using a compatible InceptionV3 head. "
    )
    if tf_err or k_err:
        msg += "\n\nErrors:\n"
        if tf_err:
            msg += f"- tf.keras: {type(tf_err).__name__}: {tf_err}\n"
        if k_err:
            msg += f"- Keras 3:  {type(k_err).__name__}: {k_err}\n"
    if loaded_from_h5:
        msg += "\nTried loading weights by layer name; any matching layers were loaded over ImageNet."
    msg += (
        "\n\nBest fix: re-export locally with tf.keras and upload `.keras` or SavedModel:\n"
        "    from tensorflow.keras.models import load_model\n"
        "    m = load_model('drosophila_inceptionv3_classifier.h5', compile=False)\n"
        "    m.save('drosophila_inceptionv3_classifier.keras')\n"
    )
    st.warning(msg)
    return model

model = load_or_fallback_model()

# ─── Inference utils ───────────────────────────────────────────────────────────
def preprocess_image(pil: Image.Image) -> np.ndarray:
    pil = pil.resize((INPUT_SIZE, INPUT_SIZE)).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32)
    return preprocess_input(arr)  # InceptionV3 expects [-1, 1]

def classify(pil: Image.Image):
    arr = preprocess_image(pil)
    preds = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(preds))
    return STAGE_LABELS[idx], float(preds[idx])

# ─── UI ────────────────────────────────────────────────────────────────────────
st.title("Live Drosophila Detection")
st.caption("Per-frame predictions. If the camera stalls, add TURN credentials in secrets (see code comment).")

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

# ─── Start Webcam with STUN/TURN ───────────────────────────────────────────────
rtc_cfg = get_rtc_configuration()
webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=SimpleProcessor,
    async_processing=False,
    rtc_configuration=rtc_cfg
)

# ─── Snapshot fallback ─────────────────────────────────────────────────────────
st.divider()
snap = st.camera_input("If live video can’t connect, use the snapshot fallback:")
if snap is not None:
    pil = Image.open(snap)
    label, conf = classify(pil)
    st.success(f"{label} ({conf:.0%})")
    st.image(pil, caption="Snapshot prediction")
# ---------------- End app.py ---------------
