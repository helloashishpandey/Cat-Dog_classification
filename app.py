import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request
from io import BytesIO

app = Flask(__name__)
model_path = 'best_model.h5'

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        image = request.files['image']
        if image:
            img = keras_image.load_img(BytesIO(image.read()), target_size=(256, 256))
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            model = load_model(model_path)
            prediction_value = model.predict(img_array)
            predicted_class_index = np.argmax(prediction_value)

            # Map the index to class labels
            class_labels = ["dog", "cat"]
            predicted_class = class_labels[predicted_class_index]

    return render_template('upload.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
