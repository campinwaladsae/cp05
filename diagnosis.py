import joblib
import tensorflow as tf
import numpy as np
from scipy.ndimage import rotate
from PIL import Image

def load_model():
    # Return the loaded model
    # For newer versions of joblib, the 'allow_pickle' argument is no longer needed
    # If you are using an older version of joblib where 'allow_pickle' is necessary, add it to the function
    return joblib.load("C:/Users/Campin Waladsae A/Downloads/cp05-main/Covid19-dataset/immune_deficiency_classifier.pkl")

def predict_diagnosis(image_array):
    model = load_model()
    prediction = model.predict(image_array)
    return "Normal" if prediction[0] == 0 else "Abnormal"

def diagnosis(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB') # Load image in RGB format
    image = image.resize((128, 128)) # Resize image to 128x128 pixels
    image = np.array(image) / 255.0 # Convert image to numpy array and normalize pixel values
    image = image.reshape(1, 128, 128, 3) # Reshape image to the required format for the model
    return predict_diagnosis(image)

def diagnosis(file_bytes):
    # Convert the image into the format that can be processed by the model
    # For example, you can convert it into a numpy array of size 1x128x128x3
    image = Image.open(file_bytes)
    image = image.resize((128, 128))
    image = np.array(image)

    # Map the prediction to the corresponding class name
    # For example, you can use a dictionary {0: 'class1', 1: 'class2', 2: 'class3'}
    return predict_diagnosis(image)