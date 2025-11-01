import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ===============================
# PATH CONFIG
# ===============================
dataset_path = "dataset_fixed"  # Change to your dataset folder
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 20
INIT_LR = 1e-3

# ===============================
# DATA AUGMENTATION
# ===============================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print(f"Class indices: {train_gen.class_indices}")

# Save class indices for app.py
with open("class_indices.pkl", "wb") as f:
    pickle.dump(train_gen.class_indices, f)
print("✅ Saved class indices as class_indices.pkl")

# ===============================
# MODEL BUILDING
# ===============================
base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze base initially

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer=Adam(learning_rate=INIT_LR), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ===============================
# CALLBACKS
# ===============================
checkpoint = ModelCheckpoint("deepfake_detection_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)

# ===============================
# CLASS WEIGHTS
# ===============================
counter = Counter(train_gen.classes)
total = sum(counter.values())
class_weight = {cls: total / (len(counter) * count) for cls, count in counter.items()}
print(f"Class weights: {class_weight}")

# ===============================
# STAGE 1 TRAINING
# ===============================
print("\n[STAGE 1] Training with frozen base layers\n")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weight,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# ===============================
# STAGE 2 FINE-TUNING
# ===============================
print("\n[STAGE 2] Fine-tuning top layers\n")
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"])

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weight,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# ===============================
# MERGE HISTORIES
# ===============================
history = {}
for key in history1.history.keys():
    history[key] = history1.history[key] + history2.history[key]

# ===============================
# PLOT TRAINING GRAPHS
# ===============================
epochs = range(len(history["loss"]))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, history["accuracy"], label="Train Accuracy", marker='o')
plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy", marker='s')
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
plt.plot(epochs, history["loss"], label="Train Loss", marker='o')
plt.plot(epochs, history["val_loss"], label="Validation Loss", marker='s')
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
plt.close()
print("✅ Saved training graphs as training_results.png")
print("✅ Training complete! Model saved as deepfake_detection_model.h5")
