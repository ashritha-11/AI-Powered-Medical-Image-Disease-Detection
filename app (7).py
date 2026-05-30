
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 AI-Powered Pneumonia Detection")
st.write("Upload a Chest X-Ray image to detect Pneumonia.")

# Load model
model = tf.keras.models.load_model("cnn_model.h5")

IMG_SIZE = 150

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_GRAYSCALE
    )

    st.image(uploaded_file, caption="Uploaded X-Ray")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(
        1,
        IMG_SIZE,
        IMG_SIZE,
        1
    ) / 255.0

    prediction = model.predict(img)[0][0]

    st.subheader("Prediction")

    if prediction > 0.5:
        st.error("⚠ Pneumonia Detected")
        st.write(f"Confidence: {prediction:.2%}")
    else:
        st.success("✅ Normal")
        st.write(f"Confidence: {(1-prediction):.2%}")
