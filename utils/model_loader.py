import tensorflow as tf
import os

breast_model = None
oral_model = None
breast_size = 224
oral_size = 224

def load_models():
    global breast_model, oral_model, breast_size, oral_size
    
    # === Breast Model ===
    breast_paths = [
        "models/breast_model.keras"
    ]
    
    for path in breast_paths:
        if os.path.exists(path):
            breast_model = tf.keras.models.load_model(path)
            print(f"✅ Breast model loaded from {path}")
            break
    else:
        print("⚠️ Breast model not found in any expected location")

    # === Oral Model ===
    oral_paths = [
        "models/oral_model.keras"
    ]
    
    for path in oral_paths:
        if os.path.exists(path):
            oral_model = tf.keras.models.load_model(path)
            print(f"✅ Oral model loaded from {path}")
            break
    else:
        print("⚠️ Oral model not found")


def predict_image(image_path, cancer_type):
    """Simple and robust prediction - Final version"""
    global breast_model, oral_model
    
    try:
        if cancer_type.lower() == "breast":
            model = breast_model
            img_size = 299                          # InceptionV3 needs 299x299
            class_names = ["Cancer", "Non Cancer"]
        else:
            model = oral_model
            img_size = 224
            class_names = ["Cancerous", "Non Cancerous"]

        if model is None:
            return f"{cancer_type.capitalize()} model not loaded", 0.0

        # Load image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_size, img_size))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Correct preprocessing for InceptionV3 (Breast model)
        if cancer_type.lower() == "breast":
            img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
        else:
            img_array = tf.keras.applications.densenet.preprocess_input(img_array)

        # Run prediction
        preds = model.predict(img_array, verbose=0)
        predicted_idx = int(tf.argmax(preds[0]))
        confidence = float(preds[0][predicted_idx]) * 100

        return class_names[predicted_idx], confidence

    except Exception as e:
        print(f"ERROR in predict_image ({cancer_type}): {str(e)}")
        return "Prediction Error", 0.0