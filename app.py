from flask import Flask, request, jsonify
import numpy as np
import joblib
from PIL import Image
import io
from tensorflow.keras.applications import ResNet50, DenseNet121, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.preprocessing.image import img_to_array
import os

app = Flask(__name__)

# ---------- Load all models ----------
model = joblib.load("plant_hybrid_stacking.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
densenet_model = DenseNet121(weights="imagenet", include_top=False, pooling="avg")
mobilenet_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

def extract_features(image, model, preprocess_func):
    image = image.resize((224, 224))
    array = img_to_array(image)
    array = np.expand_dims(array, axis=0)
    array = preprocess_func(array)
    feat = model.predict(array, verbose=0)
    return feat.flatten()

def get_fusion_features(image):
    f1 = extract_features(image, resnet_model, resnet_preprocess)
    f2 = extract_features(image, densenet_model, densenet_preprocess)
    f3 = extract_features(image, mobilenet_model, mobilenet_preprocess)
    return np.concatenate([f1, f2, f3]).reshape(1, -1)

@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Image required"}), 400

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")

    # Extract features
    features = get_fusion_features(image)
    features_scaled = scaler.transform(features)

    pred = model.predict(features_scaled)[0]
    pred_class = label_encoder.inverse_transform([pred])[0]
    probs = model.predict_proba(features_scaled)[0]

    return jsonify({
        "prediction": pred_class,
        "probabilities": {cls: float(probs[i]) for i, cls in enumerate(label_encoder.classes_)}
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
