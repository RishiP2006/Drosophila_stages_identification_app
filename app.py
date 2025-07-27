import streamlit as st
st.set_page_config(layout="centered")

# Core imports
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model as lm
# Preprocess functions
from tensorflow.keras.applications.convnext import preprocess_input as cx_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as iv3_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import time

# ─── Configuration ─────────────────────
HF_REPO_ID = "RishiPTrial/stage_modelv2"
MODELS_INFO = {
    "ConvNeXt": {"filename": "best_convnext_model_IIT.keras", "preprocess": cx_preprocess, "size": 224},
    "InceptionV3": {"filename": "drosophila_inceptionv3_classifier.h5", "preprocess": iv3_preprocess, "size": 299},
    "ResNet50": {"filename": "drosophila_stage_resnet50_finetuned_IIT.keras", "preprocess": resnet_preprocess, "size": 224},
}
CLASS_NAMES = [
    "egg", "1st instar", "2nd instar", "3rd instar",
    "white pupa", "brown pupa", "eye pupa", "black pupa",
]

# ─── Load Models ───────────────────────
@st.cache_resource
def load_models():
    models = {}
    for name, info in MODELS_INFO.items():
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=info["filename"])
        custom_objects = {"preprocess_input": info["preprocess"]}
        models[name] = (
            lm(model_path, compile=False, custom_objects=custom_objects),
            info["preprocess"],
            info["size"],
        )
    return models

models = load_models()

# ─── Prediction Helpers ────────────────
def preprocess_image(pil_img: Image.Image, size: int, preprocess_fn):
    img = pil_img.resize((size, size)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    return preprocess_fn(arr)


def classify_single(pil_img: Image.Image, model, preprocess_fn, size: int):
    arr = preprocess_image(pil_img, size, preprocess_fn)
    preds = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx])


def classify_ensemble(pil_img: Image.Image, models_dict):
    votes, confs = [], {}
    for m, fn, sz in models_dict.values():
        lbl, cf = classify_single(pil_img, m, fn, sz)
        votes.append(lbl)
        confs.setdefault(lbl, []).append(cf)
    from collections import Counter
    top = Counter(votes).most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        # tie: choose label with highest avg confidence
        avg_conf = {lbl: np.mean(lst) for lbl, lst in confs.items()}
        choice = max(avg_conf, key=avg_conf.get)
    else:
        choice = top[0][0]
    return choice, max(confs[choice])

# ─── Streamlit UI ──────────────────────
st.title("Drosophila Stage Detection")
mode = st.radio("Select Mode:", ["Single Model", "Ensemble"])

single_cfg = None
if mode == "Single Model":
    choice = st.selectbox("Choose a model:", list(models.keys()))
    single_cfg = models[choice]
else:
    st.info("Using ensemble of all three models")

# ─── Static Image Prediction ───────────
uploaded_file = st.file_uploader("Upload image for prediction", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    if mode == "Single Model" and single_cfg:
        label, conf = classify_single(img, *single_cfg)
    else:
        label, conf = classify_ensemble(img, models)
    st.image(img, caption=f"{label} ({conf:.2f})")
    st.write(f"**Prediction:** {label} | **Confidence:** {conf:.2f}")

st.markdown("---")

# ─── Live Video Prediction ─────────────
st.subheader("Live Camera Prediction")

if mode == "Single Model" and single_cfg is None:
    st.warning("Please select a model for live detection.")
else:
    # Define processor class inside this block so it captures single_cfg and mode
    class LiveProcessor(VideoProcessorBase):
        def __init__(self):
            self.mode = mode
            self.single_cfg = single_cfg
            self.models = models
            self.last_time = 0
            self.last_label = "Waiting..."
            self.last_conf = 0.0
            self.interval = 1.0  # seconds

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="rgb24")
            pil = Image.fromarray(img)

            now = time.time()
            if now - self.last_time > self.interval:
                if self.mode == "Single Model":
                    lbl, cf = classify_single(pil, *self.single_cfg)
                else:
                    lbl, cf = classify_ensemble(pil, self.models)
                self.last_label = lbl
                self.last_conf = cf
                self.last_time = now
            else:
                lbl, cf = self.last_label, self.last_conf

            # draw
            draw = ImageDraw.Draw(pil)
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
            text = f"{lbl} ({cf:.0%})"
            w = draw.textlength(text, font)
            draw.rectangle([0, 0, w + 8, 32], fill="black")
            draw.text((4, 4), text, font=font, fill="lime")
            return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

    webrtc_streamer(
        key="live_camera",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=LiveProcessor,
        async_processing=True,
    )