"""
============================================================
                    AirScript AI
------------------------------------------------------------
File        : predict.py
Author      : Anvitha K V
Description : Predict handwritten letters using the
              trained CNN model.
============================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from preprocess import preprocess_image, preprocess_canvas

# ==========================================================
# Load Model
# ==========================================================

MODEL_PATH = "models/airscript_best.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

print("=" * 60)
print("Loading AirScript AI Model...")
print("=" * 60)

model = load_model(MODEL_PATH)

print("Model loaded successfully!")

# ==========================================================
# Class Names
# ==========================================================

class_names = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

# ==========================================================
# Predict from Image File
# ==========================================================

def predict_letter(image_path):
    """
    Predict a handwritten letter from an image file.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Preprocess image
    processed = preprocess_image(image)

    # Display processed image
    plt.figure(figsize=(3, 3))
    plt.imshow(processed.squeeze(), cmap="gray")
    plt.title("Processed Image")
    plt.axis("off")
    plt.show()

    # Add batch dimension
    processed = np.expand_dims(processed, axis=0)

    # Predict
    prediction = model.predict(processed, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction) * 100)

    letter = class_names[predicted_index]

    return letter, confidence


# ==========================================================
# Predict from Air Drawing Canvas
# ==========================================================

def predict_canvas(canvas):
    """
    Predict a handwritten letter from the air-drawing canvas.
    """

    processed = preprocess_canvas(canvas)

    # Show image being sent to CNN
    plt.figure(figsize=(3, 3))
    plt.imshow(processed.squeeze(), cmap="gray")
    plt.title("Processed Canvas")
    plt.axis("off")
    plt.show()

    prediction = model.predict(processed, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction) * 100)

    letter = class_names[predicted_index]

    return letter, confidence


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    image_path = input("Enter image path: ")

    letter, confidence = predict_letter(image_path)

    print("\nPrediction")
    print("-" * 30)
    print(f"Predicted Letter : {letter}")
    print(f"Confidence       : {confidence:.2f}%")