# SmartMask_Real-Time_Face_Mask_Classifier
Face mask classification using MobileNetV2 and Streamlit

# Real-Time Face Mask Detection Web App 😷

This is a Streamlit web app that uses your webcam to detect faces and classify if the person is:
- Wearing a mask properly
- Not wearing a mask
- Wearing a mask incorrectly

## 🔧 Features

- Real-time webcam input
- Face detection + bounding box
- Mask classification using TensorFlow
- Color-coded labels for easy visual feedback

## 📦 Requirements

See `requirements.txt`. Installed automatically on Streamlit Cloud.

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app/app_realtime.py

