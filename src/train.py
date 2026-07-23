import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ==========================================================
# Random Seed
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================================
# Create Required Folders
# ==========================================================

MODEL_DIR = "models"
ASSET_DIR = "assets"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ASSET_DIR, exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading A-Z Handwritten Dataset...")
print("=" * 60)

df = pd.read_csv(
    "dataset/A_Z Handwritten Data.csv",
    header=None
)

print("Original Dataset Shape :", df.shape)

# ==========================================================
# Use 100000 Samples (for faster training)
# ==========================================================

df = df.sample(
    n=100000,
    random_state=42
).reset_index(drop=True)

print("Dataset Used :", df.shape)

# ==========================================================
# Split Features & Labels
# ==========================================================

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

print("\nFeatures Shape :", X.shape)
print("Labels Shape   :", y.shape)

# ==========================================================
# Preprocess Images
# ==========================================================

print("\nPreprocessing Images...")

X = X.astype("float32") / 255.0
X = X.reshape(-1, 28, 28, 1)

print("Processed Shape :", X.shape)

# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Shape :", X_train.shape)
print("Test Shape  :", X_test.shape)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================================================
# Build CNN Model
# ==========================================================

print("\n" + "=" * 60)
print("Building CNN Model...")
print("=" * 60)

model = Sequential([

    Input(shape=(28, 28, 1)),

    # --------------------------------------------------
    # Block 1
    # --------------------------------------------------

    Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    MaxPooling2D((2, 2)),

    Dropout(0.25),

    # --------------------------------------------------
    # Block 2
    # --------------------------------------------------

    Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    MaxPooling2D((2, 2)),

    Dropout(0.30),

    # --------------------------------------------------
    # Block 3
    # --------------------------------------------------

    Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling2D((2, 2)),

    Dropout(0.40),

    # --------------------------------------------------
    # Fully Connected Layers
    # --------------------------------------------------

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

print("\nModel Summary")
print("-" * 60)
model.summary()

# ==========================================================
# Compile Model
# ==========================================================

print("\nCompiling Model...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model compiled successfully.")

# ==========================================================
# Callbacks
# ==========================================================

print("\nCreating Callbacks...")

checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "airscript_best.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

callbacks = [
    checkpoint,
    early_stop,
    reduce_lr
]

print("Callbacks ready.")

# ==========================================================
# Train Model
# ==========================================================

print("\n" + "=" * 60)
print("Training AirScript AI Model...")
print("=" * 60)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# ==========================================================
# Evaluate Model
# ==========================================================

print("\n" + "=" * 60)
print("Evaluating Model...")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\nTest Accuracy : {test_accuracy * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# ==========================================================
# Save Final Model
# ==========================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "airscript_final.keras"
)

model.save(final_model_path)

print("\nFinal model saved successfully.")