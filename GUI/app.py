from flask import Flask, request, jsonify, render_template, send_from_directory
import model
import csv
import json
import ydf 
import pandas as pd
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('GUI', 'static', 'uploads')

model = ydf.load_model("GUI/model")
model2 = tf.keras.models.load_model("GUI/done.keras")

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('index.html')

# New endpoint to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    input_type = request.form.get('input_type')
    uploaded_file = request.files['file']

    if input_type == 'csv' and uploaded_file and allowed_file(uploaded_file.filename, {'csv'}):
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_path)

        with open(file_path, 'r') as file:
            csv_reader = csv.DictReader(file)
            data = [row for row in csv_reader]
        
        newData = pd.DataFrame(data)
        prediction = model.predict(newData)
        os.remove(file_path)

        if prediction[0] == 1.0:
            return render_template('malignant.html')
        else:
            return render_template('benign.html')

    elif input_type == 'image' and uploaded_file and allowed_file(uploaded_file.filename, {'png', 'jpg', 'jpeg', 'gif'}):
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_path)
        
        # Debug print to check file path and existence
        print(f"File saved at: {file_path}")
        if os.path.exists(file_path):
            print("File exists!")

        # Open and display the image
        image = Image.open(file_path)
        # Ensure the image has 3 channels
        image = image.convert('RGB')
        image = image.resize((224, 224))  # Resize to match the model's input shape
        image = np.array(image)  # Convert image to numpy array
        image = preprocess_input(image)  # Preprocess the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        prediction = model2.predict(image)

        if prediction[0][0] < 0.5:
            return render_template('malignant.html', filename=filename)
        else:
            return render_template('benign.html', filename=filename)

    return "Invalid input or file type"

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == "__main__":
    app.run(debug=True)
