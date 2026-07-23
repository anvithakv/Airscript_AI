"""
============================================================
                    AirScript AI
------------------------------------------------------------
File        : model.py
Author      : Anvitha K V
Description : CNN Model Architecture
============================================================
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)


def build_model():

    model = Sequential([

        Input(shape=(28, 28, 1)),

        # -----------------------------
        # Block 1
        # -----------------------------
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        MaxPooling2D((2, 2)),

        Dropout(0.25),

        # -----------------------------
        # Block 2
        # -----------------------------
        Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        MaxPooling2D((2, 2)),

        Dropout(0.30),

        # -----------------------------
        # Block 3
        # -----------------------------
        Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling2D((2, 2)),

        Dropout(0.40),

        Flatten(),

        Dense(
            256,
            activation="relu"
        ),

        Dropout(0.50),

        Dense(
            26,
            activation="softmax"
        )

    ])

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()