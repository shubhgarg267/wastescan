"""Streamlit demo for the trained aerial waste detection baseline.

Run with:
    python -m pip install -r requirements.txt
    streamlit run app.py
    python -m streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from PIL import Image

from src.domain_guard import classify_domain, load_clip_model
from src.inference import load_model, run_inference_with_model

CONFIG_PATH = "experiments/configs/baseline_yolo_gpu_full.yaml"
WEIGHTS_PATH = "experiments/checkpoints/baseline_yolo_gpu_full/weights/best.pt"


@st.cache_resource
def get_model():
    return load_model(CONFIG_PATH, WEIGHTS_PATH)


@st.cache_resource
def get_clip_model():
    return load_clip_model()


st.title("Aerial Waste Detection — Demo")
st.write(
    "Upload a top-down/aerial image to run the trained YOLOv8 baseline waste "
    "detector and see predicted bounding boxes plus a per-zone severity read."
)

conf_threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Checking image domain..."):
        domain_result = classify_domain(get_clip_model(), image)

    proceed = domain_result.is_aerial
    if not domain_result.is_aerial:
        st.warning(
            "This doesn't look like an aerial/outdoor waste-site image. The model is "
            "trained only on top-down aerial photos and may produce unreliable results "
            "on other image types."
        )
        proceed = st.checkbox("Run detection anyway")

    if proceed:
        loaded_model = get_model()

        with st.spinner("Running detection..."):
            annotated_rgb, detections, report = run_inference_with_model(
                loaded_model, image, conf_threshold=conf_threshold
            )

        st.image(annotated_rgb, caption="Detections + severity overlay")

        num_detections = len(detections)
        avg_confidence = sum(d["score"] for d in detections) / num_detections if num_detections else 0.0

        col1, col2, col3 = st.columns(3)
        col1.metric("Waste instances detected", num_detections)
        col2.metric("Average confidence", f"{avg_confidence:.2f}")
        col3.metric("Overall severity", report.overall_level.upper())
else:
    st.info("Upload an image to get started.")
