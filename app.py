import os
import numpy as np
from collections import Counter
from flask import Flask, request, render_template, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from werkzeug.utils import secure_filename
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Paths
UPLOAD_FOLDER = 'static/uploads'
MODEL_FOLDER = 'final_epochs'
CLASS_NAMES_PATH = os.path.join(MODEL_FOLDER, 'class_names.npy')
class_names = np.load(CLASS_NAMES_PATH)

# Model files
model_map = {
    "DenseNet201": "densenet201_gemstone_final_model.h5",
    "DenseNet169": "densenet169_gemstone_final_model.h5",
    "MobileNetV2": "mobilenetv2_gemstone_final.h5",
    "ConvNeXtSmall": "densenet121newdataset_80_20_final_model.h5",
    "NasnetMobile+EfficientNetB3": "best_model.h5"
}

# Load models (compile=False to speed load if you don't need training)
loaded_models = {
    model_name: load_model(os.path.join(MODEL_FOLDER, model_file), compile=False)
    for model_name, model_file in model_map.items()
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_from_image_array(img_array):
    individual_predictions = []
    voted_classes = []

    all_preds = []

    for model_name, model in loaded_models.items():
        preds = model.predict(img_array)[0]
        all_preds.append(preds)

        pred_index = np.argmax(preds)
        pred_class = class_names[pred_index]
        individual_predictions.append((model_name, pred_class, float(preds[pred_index])))
        voted_classes.append(pred_class)

    vote_counter = Counter(voted_classes)
    final_class = vote_counter.most_common(1)[0][0]

    matching_confidences = [
        conf for _, pred_cls, conf in individual_predictions if pred_cls == final_class
    ]
    confidence = float(np.mean(matching_confidences)) if matching_confidences else None

    final_pred_index = int(np.where(class_names == final_class)[0][0])
    y_true = [final_pred_index] * len(all_preds)
    y_preds = [int(np.argmax(p)) for p in all_preds]

    precision = float(precision_score(y_true, y_preds, average='macro', zero_division=0))
    recall = float(recall_score(y_true, y_preds, average='macro', zero_division=0))
    f1 = float(f1_score(y_true, y_preds, average='macro', zero_division=0))

    return final_class, confidence, precision, recall, f1, individual_predictions


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = False
    image_path = None
    confidence = None
    class_name = None
    precision = None
    recall = None
    f1 = None
    individual_predictions = []

    if request.method == 'POST':
        image_file = request.files.get('image')

        if image_file and image_file.filename != '':
            try:
                filename = secure_filename(image_file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                image_file.save(filepath)

                img = Image.open(filepath).convert('RGB')
                img_resized = img.resize((224, 224))
                img_array = keras_image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                class_name, confidence, precision, recall, f1, individual_predictions = predict_from_image_array(img_array)

                prediction = True
                image_path = filepath

            except Exception as e:
                flash(f"⚠️ Error during prediction: {e}", "danger")
        else:
            flash("⚠️ Please upload an image.", "warning")

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path,
        confidence=confidence,
        class_name=class_name,
        precision=precision,
        recall=recall,
        f1=f1,
        individual_predictions=individual_predictions
    )


if __name__ == '__main__':
    app.run(debug=True)
