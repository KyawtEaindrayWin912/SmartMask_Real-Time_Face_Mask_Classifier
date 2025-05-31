import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

MODEL_PATH = "../models/mask_classifier.h5"
IMAGE_SIZE = 224
CLASS_NAMES = ['mask_weared_incorrect', 'with_mask', 'without_mask']

@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

label_map = {
    "mask_weared_incorrect": "Incorrect Mask",
    "with_mask": "With Mask",
    "without_mask": "No Mask"
}

# Load OpenCV face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            # Extract face ROI
            face_img = img[y:y+h, x:x+w]
            # Resize to model input size
            face_resized = cv2.resize(face_img, (IMAGE_SIZE, IMAGE_SIZE))
            face_array = img_to_array(face_resized) / 255.0
            face_array = np.expand_dims(face_array, axis=0)

            # Predict mask status
            preds = model.predict(face_array)[0]
            class_id = np.argmax(preds)
            confidence = preds[class_id]
            label = CLASS_NAMES[class_id]
            pretty_label = label_map.get(label, label)

            # Draw bounding box with label
            color_map = {
                "with_mask": (0, 255, 0),         # Green
                "without_mask": (0, 0, 255),      # Red
                "mask_weared_incorrect": (0, 255, 255)  # Yellow
            }
            color = color_map.get(label, (255, 255, 255))  # default white

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = f"{pretty_label}: {confidence:.2f}"
            # Draw filled rectangle for text background
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x, y - text_height - baseline - 10), (x + text_width, y), color, -1)
            # Draw text (black font on colored bg)
            cv2.putText(img, text, (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return frame.from_ndarray(img, format="bgr24")


st.title("📷 Real-Time Mask Detection with Face Bounding Box")
webrtc_streamer(key="mask-detect-box", video_processor_factory=VideoProcessor, rtc_configuration=rtc_config)
