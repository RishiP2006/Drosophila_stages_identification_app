# Drosophila Stage — Live Video Classifier

Minimal **Streamlit** app for live webcam classification of *Drosophila* developmental stages using **Keras 3** models hosted on Hugging Face.  

**Models used (from your HF repo):**
- `drosophila_stage_resnet50_finetuned.keras`
- `best_convnext_model.keras`

---

## Run Locally

### 1. Create and activate a virtual environment  
Python 3.10 or 3.11 is recommended.


```python -m venv .venv```

Windows:
```.venv\Scripts\activate```

Mac/Linux:
```source .venv/bin/activate```

### 2. Install dependencies

``` pip install -r requirements.txt```

### 3. Start the app
``` streamlit run app.py```

Open the browser link provided by Streamlit, allow camera permissions, select a model, and you’ll see live predictions overlaid on the video feed.

### How It Works (Short)

Pure Keras 3 model loading (no tf.keras imports).

If a legacy Lambda(preprocess_input) prevents safe load, the app retries with a custom mapping and applies preprocessing externally:

ResNet50 → Caffe-style preprocessing (BGR + mean subtraction).

ConvNeXt → Simple normalization (divide by 255.0).

### Troubleshooting

Black/frozen preview → ensure HTTPS, allow camera permissions, refresh the page.

Mobile support → live video feature currently works best on mobile devices.
