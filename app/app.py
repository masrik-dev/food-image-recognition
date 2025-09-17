import os
import base64
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image
from io import BytesIO

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), "final_model.keras")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = load_model(model_path)

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("label_classes.npy")


def preprocess_image(img, img_size=(224, 224)):
    img = img.resize(img_size)
    img_array = np.array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

def predict_food_type_with_confidence(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_label_idx = np.argmax(predictions, axis=1)[0]
    confidence =predictions[0][predicted_label_idx]
    food_type = label_encoder.inverse_transform([predicted_label_idx])
    return food_type[0], confidence

def encode_image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

def resize_image_if_needed(img, max_size=(750, 750)):
    if img.width > max_size[0] or img.height > max_size[1]:
        img.thumbnail(max_size, Image.LANCZOS)
    return img

@app.route("/", methods=["GET", "POST"])
def upload_image():
    result_image = None
    result_text = None
    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                try:
                    img = Image.open(file.strem)
                    img = resize_image_if_needed(img)
                    food_type, cofidence = predict_food_type_with_confidence(img)
                    result_text = f"{food_type} ({confidence: .2%})"
                    result_image = encode_image_to_base64(img)
                except Exception as e:
                    print(f"Error processing image: {e}")
                finally:
                    file.stream.close()
                    img.close()
    return render_template("index.html", result_image=result_image, result_text=result_text)

if __name__ == "__main__":
    app.run(debug=True)