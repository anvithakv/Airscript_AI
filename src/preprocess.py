"""
============================================================
                    AirScript AI
------------------------------------------------------------
File        : preprocess.py
Author      : Anvitha K V
Description : Image preprocessing utilities for training
              and real-time prediction.
============================================================
"""

import cv2
import numpy as np


# ==========================================================
# Resize Image
# ==========================================================

def resize_image(image, size=(28, 28)):
    """
    Resize image to the required input size.

    Parameters:
        image (numpy.ndarray): Input image.
        size (tuple): Target size.

    Returns:
        numpy.ndarray
    """
    return cv2.resize(image, size)


# ==========================================================
# Normalize Image
# ==========================================================

def normalize_image(image):
    """
    Normalize pixel values between 0 and 1.
    """
    image = image.astype("float32")
    image /= 255.0
    return image


# ==========================================================
# Center Character
# ==========================================================

def center_character(image):
    """
    Crop, resize and center the handwritten character.
    """

    coordinates = cv2.findNonZero(image)

    if coordinates is None:
        return np.zeros((28, 28), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coordinates)

    cropped = image[y:y+h, x:x+w]

    # Resize while keeping aspect ratio
    if h > w:
        new_h = 20
        new_w = max(1, int(w * 20 / h))
    else:
        new_w = 20
        new_h = max(1, int(h * 20 / w))

    resized = cv2.resize(
    cropped,
    (new_w, new_h),
    interpolation=cv2.INTER_AREA
)
    kernel = np.ones((2, 2), np.uint8)
    resized = cv2.dilate(resized, kernel, iterations=1)

    canvas = np.zeros((28, 28), dtype=np.uint8)

    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2

    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized

    return canvas

# ==========================================================
# Preprocess Image
# ==========================================================

def preprocess_image(image):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binary threshold (white background -> black, letter -> white)
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Center and resize
    processed = center_character(thresh)

    # Normalize
    processed = processed.astype("float32") / 255.0

    # Add channel dimension
    processed = processed.reshape(28, 28, 1)

    return processed

# ==========================================================
# Preprocess Canvas
# ==========================================================

def preprocess_canvas(canvas):
    """
    Preprocess the drawing canvas for CNN prediction.

    Parameters:
        canvas (numpy.ndarray): OpenCV drawing canvas.

    Returns:
        numpy.ndarray: Shape (1, 28, 28, 1)
    """

    # Convert canvas to grayscale
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # Center the handwritten character
    processed = center_character(gray)

    # Normalize
    processed = normalize_image(processed)

    # Add channel dimension
    processed = processed.reshape(28, 28, 1)

    # Add batch dimension
    processed = np.expand_dims(processed, axis=0)

    return processed

# ==========================================================
# Batch Preprocessing
# ==========================================================

def preprocess_batch(images):
    """
    Preprocess multiple images.
    """

    processed = []

    for image in images:
        processed.append(
            preprocess_image(image)
        )

    return np.array(processed)


# ==========================================================
# Main (Testing)
# ==========================================================

if __name__ == "__main__":

    image = cv2.imread("sample.png")

    if image is None:
        print("Sample image not found.")
    else:
        processed = preprocess_image(image)

        print("Image processed successfully.")
        print(processed.shape)