from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model("leaf_classifier_model.h5")
class_names = ['Aloevera', 'Amla', 'Amruthaballi','Arali','ashoka','Astma_weed','Badipala','Balloon_Vine','Bamboo','Beans','Betel','Bhrami','Bringraja','camphor','Caricature','Castor','Catharanthus','Chakte','Chilly']  # Replace with actual labels

def preprocess_image(image):
    image = image.resize((299, 299))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            processed = preprocess_image(image)
            prediction = model.predict(processed)
            label = class_names[np.argmax(prediction)]
            
            # Image path based on predicted label
            image_path = f"images/{label}.png"  # Looks inside /static/images/
            
            return render_template("index.html", prediction=label, image_path=image_path)
    return render_template("index.html", prediction=None, image_path=None)


if __name__ == "__main__":
    app.run(debug=True)
