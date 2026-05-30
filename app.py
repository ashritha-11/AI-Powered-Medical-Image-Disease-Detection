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
# MODEL CONFIG
# =====================================

MODEL_FILE = "cnn_model.h5"
FILE_ID = "1I5V92KD5efDyJOrUdKS0F4iPmKnPGFnd"

# =====================================
# DOWNLOAD MODEL
# =====================================

@st.cache_resource
def download_model():

    if not os.path.exists(MODEL_FILE):

        url = f"https://drive.google.com/uc?id={FILE_ID}"

        with st.spinner("Downloading AI model..."):
            gdown.download(url, MODEL_FILE, quiet=False)

download_model()

# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_model():

    try:
        model = tf.keras.models.load_model(
            MODEL_FILE,
            compile=False
        )
        return model

    except Exception as e:
        st.error("❌ Model Loading Failed")
        st.error(str(e))
        return None

model = load_model()

if model is None:
    st.stop()

# =====================================
# IMAGE SIZE
# =====================================

IMG_SIZE = 150

# =====================================
# UI
# =====================================

st.title("🫁 AI-Powered Pneumonia Detection")

st.markdown("""
Upload a chest X-ray image and let the CNN model predict whether pneumonia is present.
""")

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

    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    with st.spinner("Analyzing X-Ray..."):

        prediction = model.predict(
            image,
            verbose=0
        )

        score = float(prediction[0][0])

    st.subheader("Prediction")

    if score > 0.5:

        st.error("⚠️ Pneumonia Detected")
        st.metric(
            "Confidence",
            f"{score * 100:.2f}%"
        )

    else:

        st.success("✅ Normal")
        st.metric(
            "Confidence",
            f"{(1 - score) * 100:.2f}%"
        )
