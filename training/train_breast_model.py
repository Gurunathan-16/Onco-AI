import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

# -------------------------------------------------
# 🔹 ABSOLUTE PATHS (UPDATE IF NEEDED)
# -------------------------------------------------

DATASET_PATH = r"D:\onco-ai\dataset\breast_cancer"
MODEL_SAVE_PATH = r"D:\onco-ai\models\breast_model.h5"

# Create models folder if not exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# -------------------------------------------------
# Parameters
# -------------------------------------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# -------------------------------------------------
# Data Generator with Validation Split
# -------------------------------------------------

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# -------------------------------------------------
# Build Model (ResNet50 Transfer Learning)
# -------------------------------------------------

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

# -------------------------------------------------
# Compile Model
# -------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------------------------
# Train Model
# -------------------------------------------------

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# -------------------------------------------------
# Save Model
# -------------------------------------------------

model.save(MODEL_SAVE_PATH)

print("\n✅ Breast Model Saved Successfully!")
print("📁 Saved at:", MODEL_SAVE_PATH)