# ------------------ app.py ------------------
# Force CPU + quiet TF logs BEFORE any TF/Keras import
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import streamlit as st
st.set_page_config(layout="centered")

import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Keras 3 (native loader for .keras)
from keras.models import load_model as k3_load_model
from keras.applications.inception_v3 import preprocess_input

# ─── Config ────────────────────────────────────────────────────────────────────
HF_REPO_ID   = "RishiPTrial/my-model-name"
MODEL_FILE   = "drosophila_inceptionv3_classifier.keras"  # <-- use the new .keras
INPUT_SIZE   = 299
STAGE_LABELS = ["egg", "1st instar", "2nd instar", "3rd instar", "white pupa", "brown pupa", "eye pupa"]

# ─── ICE / TURN helpers ────────────────────────────────────────────────────────
def get_rtc_configuration():
    """
    For cloud use, add TURN creds in .streamlit/secrets.toml:
      [ice]
      policy = "relay"
      servers = [
        {urls = ["stun:stun.l.google.com:19302"]},
        {urls = ["turns:YOUR_TURN:5349","turn:YOUR_TURN:3478?transport=tcp"], username="USER", credential="PASS"}
      ]
    """
    ice = st.secrets.get("ice", None)
    if ice and "servers" in ice and ice["servers"]:
        cfg = {"iceServers": ice["servers"]}
        if "policy" in ice:
            cfg["iceTransportPolicy"] = ice["policy"]
        return cfg
    # default to STUN-only (fine for localhost/LAN)
    return {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

def has_turn(cfg: dict) -> bool:
    for srv in cfg.get("iceServers", []):
        urls = srv.get("urls", [])
        if isinstance(urls, str):
            urls = [urls]
        for u in urls:
            if isinstance(u, str) and u.startswith(("turn:", "turns:")):
                return True
    return False

# ─── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model (.keras) from Hugging Face…")
def load_model():
    mp = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE, token=st.secrets.get("HF_TOKEN"))
    return k3_load_model(mp, compile=False)

model = load_model()

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

def draw_overlay(pil: Image.Image, text: str):
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    try:
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    except Exception:
        w, h = draw.textsize(text, font=font)
        x0, y0, x1, y1 = 0, 0, w, h
    pad = 6
    draw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill="black")
    draw.text((0, 0), text, font=font, fill="red")
    return pil

# ─── UI ────────────────────────────────────────────────────────────────────────
st.title("Live Drosophila Detection")
st.caption("Per-frame predictions using your .keras model. Choose an input method below.")

mode = st.radio(
    "Input",
    ["Live webcam (WebRTC)", "Snapshot (no WebRTC)", "Upload video (no WebRTC)"],
    index=0,
    horizontal=True,
)

# ─── Live webcam (only start if TURN exists or user confirms localhost) ───────
if mode == "Live webcam (WebRTC)":
    rtc_cfg = get_rtc_configuration()
    if not has_turn(rtc_cfg):
        st.info("No TURN is configured. On cloud this may fail. If you are on localhost/LAN, tick below to proceed with STUN-only.")
        proceed = st.checkbox("I'm on localhost/LAN → proceed with STUN-only", value=False)
        if not proceed:
            st.stop()

    class SimpleProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="rgb24")
            pil = Image.fromarray(img)
            try:
                label, conf = classify(pil)
            except Exception:
                label, conf = "error", 0.0
            text = f"{label} ({conf:.0%})"
            pil = draw_overlay(pil, text)
            return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

    webrtc_streamer(
        key="live",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 15, "max": 15}},
            "audio": False,
        },
        video_processor_factory=SimpleProcessor,
        async_processing=False,
        rtc_configuration=rtc_cfg,
    )

# ─── Snapshot fallback (always works) ─────────────────────────────────────────
elif mode == "Snapshot (no WebRTC)":
    snap = st.camera_input("Take a snapshot:")
    if snap is not None:
        pil = Image.open(snap)
        label, conf = classify(pil)
        st.success(f"{label} ({conf:.0%})")
        st.image(draw_overlay(pil, f"{label} ({conf:.0%})"), caption="Snapshot prediction")

# ─── Video upload (simulated live) ─────────────────────────────────────────────
else:  # Upload video (no WebRTC)
    vid = st.file_uploader("Upload a short video (MP4/MOV/M4V):", type=["mp4", "mov", "m4v"])
    if vid is not None:
        placeholder = st.empty()
        with av.open(vid) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 15.0
            for frame in container.decode(video=0):
                pil = frame.to_image().convert("RGB")
                label, conf = classify(pil)
                out = draw_overlay(pil.copy(), f"{label} ({conf:.0%})")
                placeholder.image(out, use_column_width=True)
                time.sleep(1.0 / max(fps, 1.0))
# ---------------- End app.py ---------------
