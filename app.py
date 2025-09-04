# app.py
import time
import streamlit as st
st.set_page_config(layout="centered")

# Shim for older streamlit-webrtc
if not hasattr(st, "experimental_rerun") and hasattr(st, "rerun"):
    try:
        st.experimental_rerun = st.rerun  # type: ignore
    except Exception:
        pass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from huggingface_hub import hf_hub_download

# KERAS 3
from keras.models import load_model as k_load_model
from keras.applications.inception_v3 import preprocess_input

# ── App config ──────────────────────────────────────────────────────────────────
HF_REPO_ID = "RishiPTrial/my-model-name"
MODEL_FILE = "drosophila_inceptionv3_classifier.h5"
INPUT_SIZE = 299
STAGE_LABELS = ["egg","1st instar","2nd instar","3rd instar","white pupa","brown pupa","eye pupa"]

# Media constraints: small & steady
MEDIA_STREAM_CONSTRAINTS = {
    "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 15}},
    "audio": False,
}

# TURN/STUN config. Toggle "Force TURN only" to work behind strict NATs/firewalls.
BASE_ICE_SERVERS = [
    {"urls": ["stun:stun.l.google.com:19302"]},
    {
        "urls": [
            "turn:openrelay.metered.ca:80",
            "turn:openrelay.metered.ca:443",
            "turn:openrelay.metered.ca:443?transport=tcp",
        ],
        "username": "openrelayproject",
        "credential": "openrelayproject",
    },
]

# ── Model ───────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model from Hugging Face…")
def load_model():
    token = st.secrets.get("HF_TOKEN", None)
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE, token=token)
    return k_load_model(model_path, compile=False)

model = load_model()

def preprocess_image(pil: Image.Image) -> np.ndarray:
    pil = pil.resize((INPUT_SIZE, INPUT_SIZE)).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32)
    return preprocess_input(arr)

def classify(pil: Image.Image):
    arr = preprocess_image(pil)
    preds = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(preds))
    return STAGE_LABELS[idx], float(preds[idx]), preds

def draw_legend(pil: Image.Image, text: str, font):
    out = pil.copy()
    d = ImageDraw.Draw(out)
    try:
        bbox = d.textbbox((0, 0), text, font=font); w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        w, h = d.textsize(text, font=font)
    pad = 6
    d.rectangle([0-pad, 0-pad, w+pad, h+pad], fill="black")
    d.text((0, 0), text, font=font, fill="red")
    return out

# ── UI ──────────────────────────────────────────────────────────────────────────
st.title("Live Drosophila Detection")
st.caption("If the camera keeps buffering: enable 'Force TURN only' below, or try another network.")

with st.sidebar:
    st.subheader("Connection options")
    force_turn_only = st.checkbox("Force TURN only (relay)", value=True,
                                  help="Use relay only (no STUN). Fixes strict NAT/firewall issues.")
    show_debug = st.checkbox("Show WebRTC debug", value=False)

# Build RTC configuration from options
rtc_conf = {
    "iceServers": BASE_ICE_SERVERS,
    # When forced, this prevents direct (P2P) attempts that stall on strict networks.
    "iceTransportPolicy": "relay" if force_turn_only else "all",
}

# ── Video processor (no Streamlit calls here) ───────────────────────────────────
class StableProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_label = None
        self.count = 0
        self.stable_label = None
        self.stable_conf = 0.0
        self.last_infer_t = 0.0
        self.infer_interval_s = 0.4
        try:
            self.font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        except Exception:
            self.font = ImageFont.load_default()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)

        t = time.time()
        text = "…"
        if (t - self.last_infer_t) >= self.infer_interval_s:
            self.last_infer_t = t
            label, conf, _ = classify(pil)
            if label == self.last_label:
                self.count += 1
            else:
                self.last_label = label
                self.count = 1
            if self.count >= 3:
                self.stable_label, self.stable_conf = label, conf
            text = f"{label} ({conf:.0%})"
        elif self.last_label:
            text = self.last_label

        annotated = draw_legend(pil, text, self.font)
        return av.VideoFrame.from_ndarray(np.array(annotated), format="rgb24")

# ── Start Webcam (sync processing; add video attrs for autoplay/ios) ────────────
webrtc_ctx = webrtc_streamer(
    key=f"live-{'relay' if force_turn_only else 'all'}",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
    rtc_configuration=rtc_conf,
    video_processor_factory=StableProcessor,
    async_processing=False,  # more stable on Cloud
    sendback_audio=False,
    video_html_attrs={"autoPlay": True, "playsinline": True, "muted": True, "controls": True},
)

# Optional: debug panel (helps confirm ICE/connection)
if show_debug and webrtc_ctx:
    st.sidebar.write("State:", webrtc_ctx.state)
    pc = getattr(webrtc_ctx, "pc", None) or getattr(webrtc_ctx, "peer_connection", None)
    st.sidebar.write("PeerConnection present:", bool(pc))

# ── Upload path ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🖼️ Upload an Image for Classification")
uploaded = st.file_uploader("Upload a Drosophila image (JPG/PNG)", type=["jpg","jpeg","png"])
if uploaded is not None:
    from PIL import Image  # ensure imported
    pil_in = Image.open(uploaded).convert("RGB")
    label, conf, probs = classify(pil_in)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Prediction")
        st.write(f"**Stage:** {label}")
        st.write(f"**Confidence:** {conf:.2%}")
    with c2:
        st.subheader("Annotated Preview")
        st.image(draw_legend(pil_in, f"{label} ({conf:.0%})", ImageFont.load_default()), use_column_width=True)
    with st.expander("Show class probabilities"):
        rows = sorted([(lab, float(p)) for lab, p in zip(STAGE_LABELS, probs)],
                      key=lambda x: x[1], reverse=True)
        for lab, p in rows:
            st.write(f"{lab}: {p:.2%}")
