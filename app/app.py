import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# === CONFIG ===
MODEL_PATH = "../models/mask_classifier.h5"
CLASS_NAMES = ['mask_weared_incorrect', 'with_mask', 'without_mask']
IMAGE_SIZE = 224

# === Load Model ===
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# === App Header ===
st.title("😷 Mask Detection Classifier")
st.write("Upload a face image to check if the person is wearing a mask properly.")

# === File Upload ===
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)[0]
    class_id = np.argmax(preds)
    confidence = preds[class_id]
    label = CLASS_NAMES[class_id]

    # Beautify label
    label_map = {
        "mask_weared_incorrect": "Incorrect Mask",
        "with_mask": "With Mask",
        "without_mask": "No Mask"
    }

    pretty_label = label_map.get(label, label)

    # Display prediction
    st.markdown(f"### 🧠 Prediction: `{pretty_label}`")
    st.markdown(f"### ✅ Confidence: `{confidence:.2f}`")
