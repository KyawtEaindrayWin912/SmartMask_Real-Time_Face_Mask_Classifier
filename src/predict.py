import tensorflow as tf
import numpy as np
import sys
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# === CONFIG ===
IMAGE_SIZE = 224
MODEL_PATH = "../models/mask_classifier.h5"
CLASS_NAMES = ['mask_weared_incorrect', 'with_mask', 'without_mask']

# === LOAD MODEL ===
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# === LOAD AND PREPROCESS IMAGE ===
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim
    return img_array

# === PREDICT FUNCTION ===
def predict_mask(img_path):
    img_array = load_and_preprocess_image(img_path)
    preds = model.predict(img_array)[0]
    class_id = np.argmax(preds)
    confidence = preds[class_id]

    print(f"\n🧠 Prediction: {CLASS_NAMES[class_id]}")
    print(f"✅ Confidence: {confidence:.2f}")

    # Optional: display image
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{CLASS_NAMES[class_id]} ({confidence:.2f})")
    plt.show()

# === MAIN EXECUTION ===
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/image.jpg")
        sys.exit()

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print("❌ Image file not found!")
        sys.exit()

    predict_mask(image_path)
