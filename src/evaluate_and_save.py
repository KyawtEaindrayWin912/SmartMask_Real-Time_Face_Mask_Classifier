import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# === CONFIG ===
IMAGE_SIZE = 224
MODEL_PATH = "../models/mask_classifier.h5"
DATA_DIR = "../data"  # your flat structure
RESULTS_DIR = "../results"
CLASS_NAMES = ['mask_weared_incorrect', 'with_mask', 'without_mask']

# === Ensure /results folder exists ===
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Load the trained model ===
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# === Predict and save result images ===
for label in CLASS_NAMES:
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.exists(class_dir):
        print(f"⚠️ Skipping missing folder: {class_dir}")
        continue

    img_names = os.listdir(class_dir)[:10]  # adjust number if needed

    for img_name in img_names:
        img_path = os.path.join(class_dir, img_name)
        img = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)[0]
        class_id = np.argmax(preds)
        confidence = preds[class_id]
        pred_label = CLASS_NAMES[class_id]

        # Draw prediction
        VISUAL_SIZE = 512  # Increase display size
        img_cv = cv2.imread(img_path)
        img_cv = cv2.resize(img_cv, (VISUAL_SIZE, VISUAL_SIZE))
        label_text = f"{pred_label} ({confidence:.2f})"
        cv2.putText(img_cv, label_text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save to results
        save_name = f"{pred_label}_{img_name}"
        save_path = os.path.join(RESULTS_DIR, save_name)
        cv2.imwrite(save_path, img_cv)

print(f"✅ Done! Predictions saved to {RESULTS_DIR}")
