import base64
import io
import os
import pickle
from typing import List

import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)
MODEL = None
CLASS_NAMES: List[str] = [str(i) for i in range(1000)]


def load_class_names(metadata_path: str):
    global CLASS_NAMES
    if not os.path.exists(metadata_path):
        return
    with open(metadata_path, 'rb') as metadata_file:
        metadata = pickle.load(metadata_file, encoding='bytes')
    label_names = metadata.get(b'label_names', metadata.get('label_names'))
    if label_names:
        CLASS_NAMES = [
            name.decode('utf-8') if isinstance(name, bytes) else str(name)
            for name in label_names
        ]


def load_model(model_path: str):
    global MODEL
    if os.path.exists(model_path):
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        MODEL = session


def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((32, 32))
    image = np.asarray(img, dtype=np.float32) / 255.0
    image = (image - np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)) \
        / np.array([0.247, 0.243, 0.261], dtype=np.float32)
    return np.transpose(image, (2, 0, 1))[None, ...]


@app.route('/health', methods=['GET'])
def health():
    if MODEL is None:
        return 'model not loaded', 503
    return 'ok', 200


@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'model not loaded'}), 503
    model = MODEL

    try:
        uploaded_image = request.files.get('image')
        if uploaded_image is not None:
            img_bytes = uploaded_image.read()
        else:
            data = request.get_json(silent=True)
            b64 = data.get('image') if data is not None else request.form.get('image')
            if b64 is None:
                return jsonify({'error': 'missing image field (base64 or file)'}), 400
            img_bytes = base64.b64decode(b64)

        x = preprocess_image(img_bytes)
        logits = model.run(None, {model.get_inputs()[0].name: x})[0][0]
        logits = logits - np.max(logits)
        probs = (np.exp(logits) / np.exp(logits).sum()).tolist()
        pred = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else str(pred)
        return jsonify({'probs': probs, 'pred': pred, 'pred_name': pred_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    model_path = os.environ.get('MODEL_PATH', os.environ.get('MODEL_CHECKPOINT', 'checkpoints/model.onnx'))
    metadata_path = os.environ.get('METADATA_PATH', 'data/cifar-10-batches-py/batches.meta')
    load_class_names(metadata_path)
    if os.path.exists(model_path):
        load_model(model_path)
    app.run(host='0.0.0.0', port=8080)
