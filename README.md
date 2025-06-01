# SmartMask: Real-Time & Image-Based Face Mask Classifier

Face mask classification using MobileNetV2 and Streamlit.

---

## Overview

This project contains two Streamlit web apps for face mask detection:

- **Image Classification App** (`app/app.py`): Upload an image to classify whether a person is wearing a mask properly, not wearing a mask, or wearing it incorrectly.
- **Real-Time Webcam App** (`app/realtime_app.py`): Use your webcam for live face mask detection with face bounding boxes and color-coded labels.

Both apps use a MobileNetV2-based TensorFlow model for classification.

---

## Features

- Real-time webcam input for live mask detection  
- Image upload classification  
- Face detection with bounding boxes  
- Mask classification with clear, color-coded labels  
- Simple and intuitive user interface  

---


## 📦 Requirements

See `requirements.txt`. Installed automatically on Streamlit Cloud.

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app/realtime_app.py
streamlit run app/app.py

