import streamlit as st
import numpy as np
import cv2
import os
import gdown
import tensorflow as tf

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# =====================================
# MODEL DOWNLOAD
# =====================================

MODEL_FILE = "cnn_model.h5"

@st.cache_resource
def download_model():

    if not os.path.exists(MODEL_FILE):

        file_id = "1I5V92KD5efDyJOrUdKS0F4iPmKnPGFnd"

        url = f"https://drive.google.com/uc?id={file_id}"

        with st.spinner("Downloading AI Model..."):
            gdown.download(url, MODEL_FILE, quiet=False)

download_model()

# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_FILE)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# =====================================
# IMAGE SETTINGS
# =====================================

IMG_SIZE = 150

# =====================================
# HEADER
# =====================================

st.title("🫁 AI-Powered Pneumonia Detection")

st.markdown("""
Upload a **Chest X-Ray Image** and let the AI determine whether the patient has:

- ✅ Normal Lungs
- ⚠️ Pneumonia

Powered by Deep Learning (CNN)
""")

# =====================================
# FILE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "Upload X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# PREDICTION
# =====================================

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded X-Ray",
        use_container_width=True
    )

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_GRAYSCALE
    )

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = image / 255.0

    image = image.reshape(
        1,
        IMG_SIZE,
        IMG_SIZE,
        1
    )

    with st.spinner("Analyzing X-Ray..."):
        prediction = model.predict(
            image,
            verbose=0
        )[0][0]

    st.subheader("Prediction Result")

    if prediction > 0.5:

        confidence = prediction * 100

        st.error("⚠️ Pneumonia Detected")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    else:

        confidence = (1 - prediction) * 100

        st.success("✅ Normal")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )
