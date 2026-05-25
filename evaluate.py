import tensorflow as tf
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.models.base_model import create_base_model
from src.data.preprocessing import get_preprocess_function

# Load config
with open('configs/breast_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config.get('model_name', 'densenet121')
num_classes = config.get('num_classes', 2)
image_size = config.get('image_size', 299 if model_name == 'densenet121' else 224)

print(f"Evaluating {model_name.upper()} model on test set...")

# Load test dataset (without map first to get class_names)
raw_test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset_split/breast/test",
    image_size=(image_size, image_size),
    batch_size=32,
    label_mode='categorical',
    shuffle=False
)

class_names = raw_test_ds.class_names
print(f"Test classes: {class_names}")

# Now apply preprocessing
preprocess_fn = get_preprocess_function(model_name)
test_ds = raw_test_ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# Load best model
model_path = f"experiments/breast_{model_name}/best_model.keras"
if not tf.io.gfile.exists(model_path):
    model_path = f"experiments/breast_{model_name}/final_model.keras"
    print(f"Using final model: {model_path}")

model = tf.keras.models.load_model(model_path)

# Get predictions
print("\nRunning predictions on test set...")
y_true = []
y_pred = []
y_pred_prob = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred_prob.extend(preds)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred_prob = np.array(y_pred_prob)

# === Evaluation Results ===
print("\n" + "="*70)
print("BREAST CANCER DETECTION - EVALUATION RESULTS")
print("="*70)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Breast Cancer Detection')
plt.tight_layout()
plt.savefig('results/visualizations/breast_confusion_matrix.png', dpi=300)
plt.show()

# AUC Score
if num_classes == 2:
    auc = roc_auc_score(y_true, y_pred_prob[:, 1])
    print(f"\nAUC Score: {auc:.4f}")
else:
    auc = roc_auc_score(y_true, y_pred_prob, multi_class='ovr', average='macro')
    print(f"\nMacro AUC Score: {auc:.4f}")

print(f"\n✅ Confusion matrix saved to: results/visualizations/breast_confusion_matrix.png")
print("Project is working well!")