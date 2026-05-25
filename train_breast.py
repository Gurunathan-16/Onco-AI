import tensorflow as tf
import yaml
from src.data.preprocessing import get_preprocess_function, get_augmentation_layer

# ==================== LOAD CONFIG ====================
with open('configs/breast_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config.get('model_name', 'densenet121')
num_classes = config.get('num_classes', 2)
image_size = config.get('image_size', 224)
batch_size = config.get('batch_size', 32)
epochs = config.get('epochs', 3)
lr = config.get('learning_rate', 0.0001)

print(f"🚀 Training {model_name.upper()} for Breast Cancer Detection")
print(f"Image Size: {image_size} | Epochs: {epochs}")

# ==================== LOAD DATASETS ====================
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset_split/breast/train",
    image_size=(image_size, image_size),
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset_split/breast/val",
    image_size=(image_size, image_size),
    batch_size=batch_size,
    label_mode='categorical'
)

print(f"Classes: {train_ds.class_names}")

# ==================== AUGMENTATION & PREPROCESSING ====================
aug_layer = get_augmentation_layer(image_size)
preprocess_fn = get_preprocess_function(model_name)

train_ds = train_ds.map(lambda x, y: (preprocess_fn(aug_layer(x)), y), num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# ==================== CREATE MODEL ====================
from src.models.base_model import create_base_model

model = create_base_model(model_name, num_classes, 
                         input_shape=(image_size, image_size, 3), 
                         trainable=False)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==================== CALLBACKS ====================
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=f"experiments/breast_{model_name}/best_model.keras",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

# ==================== STEP 3: TRAINING WITH FINE-TUNING ====================
print("\n=== Phase 1: Training Top Layers (Classifier Head) ===")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,                    # First phase: fewer epochs
    callbacks=callbacks
)

print("\n=== Phase 2: Fine-Tuning (Unfreezing Last Layers) ===")
# Unfreeze the base model partially
base_model = model.layers[0]
base_model.trainable = True

# Freeze all layers except the last 40 (adjust this number based on model)
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Re-compile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr / 10),   # Very important: lower LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,          # Remaining epochs for fine-tuning
    callbacks=callbacks
)

print("\n✅ Training Completed Successfully!")
model.save(f"experiments/breast_{model_name}/final_model.keras")
print(f"Best model saved in: experiments/breast_{model_name}/")