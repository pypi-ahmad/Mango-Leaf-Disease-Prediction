import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

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

# Function to preprocess and classify an image
def classify_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
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
    
    # Display the image
    img = Image.open(file_path)
    img = img.resize((500, 500))
    img = ImageTk.PhotoImage(img)
    img_label.config(image=img)
    img_label.image = img
    
    # Display the prediction result
    result_label.config(text=f"Predicted Class: {predicted_class}")
    prob_label.config(text=f"Prediction Probabilities: {predictions[0]}")

# Create the main application window
app = tk.Tk()
app.title("Mango Leaf Disease Classification")
app.geometry("1366x768")

# Create buttons and labels
browse_button = tk.Button(app, text="Browse Image", command=classify_image)
img_label = tk.Label(app)
result_label = tk.Label(app, text="", font=("Helvetica", 16))
prob_label = tk.Label(app, text="", font=("Helvetica", 14))

# Place widgets on the window
browse_button.pack(pady=20)
img_label.pack()
result_label.pack()
prob_label.pack()

# Start the Tkinter main loop
app.mainloop()