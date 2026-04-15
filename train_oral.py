import tensorflow as tf
import yaml
from src.data.preprocessing import get_preprocess_function, get_augmentation_layer

# Load config
with open('configs/oral_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config.get('model_name', 'densenet121')
num_classes = config.get('num_classes', 2)
image_size = config.get('image_size', 224)
batch_size = config.get('batch_size', 32)
epochs = config.get('epochs', 25)
lr = config.get('learning_rate', 0.0001)

print(f"🚀 Training {model_name.upper()} for Oral Cancer Detection (2 classes)")

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset_split/oral/train",
    image_size=(image_size, image_size),
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset_split/oral/val",
    image_size=(image_size, image_size),
    batch_size=batch_size,
    label_mode='categorical'
)

print(f"Classes: {train_ds.class_names}")

# Augmentation & Preprocessing
aug_layer = get_augmentation_layer(image_size)
preprocess_fn = get_preprocess_function(model_name)

train_ds = train_ds.map(lambda x, y: (preprocess_fn(aug_layer(x)), y), num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# Create model (using base_model.py from breast)
from src.models.base_model import create_base_model

model = create_base_model(model_name, num_classes, input_shape=(image_size, image_size, 3), trainable=False)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=f"experiments/oral_{model_name}/best_model.keras",
        monitor='val_accuracy', save_best_only=True, verbose=1
    )
]

print("\nStarting training for Oral Cancer...\n")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
)

print("\n✅ Oral Cancer training completed!")
model.save(f"experiments/oral_{model_name}/final_model.keras")