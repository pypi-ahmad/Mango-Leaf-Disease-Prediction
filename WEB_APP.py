import os
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("mango_leaf_disease_model.h5")

# Define class labels
class_labels = {
    0: "Anthracnose",
    1: "Bacterial Canker",
    2: "Cutting Weevil",
    3: "Die Back",
    4: "Gall Midge",
    5: "Healthy",
    6: "Powdery Mildew",
    7: "Sooty Mould"
}

# Configure a folder for uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess and classify an image
def classify_image(file_path):
    # Load and preprocess the image
    img = Image.open(file_path)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make predictions
    predictions = model.predict(img)
    
    # Get the predicted class and corresponding label
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_idx]
    
    return predicted_class, predictions[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        
        # Check if the file has a valid name and extension
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Classify the uploaded image
            predicted_class, prediction_probabilities = classify_image(file_path)
            
            return render_template('result.html', 
                                   image_path=file_path, 
                                   predicted_class=predicted_class, 
                                   probabilities=prediction_probabilities)
    
    return render_template('index.html', error=None)

if __name__ == '__main__':
    app.run(debug=True)
