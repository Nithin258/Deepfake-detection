import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
IMG_SIZE = 128

model = tf.keras.models.load_model("deepfake_model.keras")
model = tf.keras.models.load_model("deepfake_model.h5")

# Title
st.title("Deepfake Detection App")

st.write("Upload an image to check if it's REAL or FAKE")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Show image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # Predict
    prediction = model.predict(img_resized)[0][0]

    # Output
    if prediction > 0.5:
        st.error(f"FAKE IMAGE ❌ (Confidence: {prediction:.2f})")
    else:

        st.success(f"REAL IMAGE ✅ (Confidence: {prediction:.2f})")

