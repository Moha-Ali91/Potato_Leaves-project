from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Initialize the FastAPI app
app = FastAPI()

# Load the pre-trained model
model = load_model('potato_model.keras')

@app.get("/")
async def read_root():

    return {
        "message": "Welcome to Potato disease classification API!",
        "instructions": {
            "POST /predict/": "Upload RGB image of a Potato (128x128 pixels) to get the predicted class."
        }
    }

# Define class names
class_names = ["Healthy", "Early Blight", "Late Blight"]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    # Convert the uploaded image to RGB, resize to 128x128 pixels
    img = Image.open(file.file).convert('RGB').resize((128, 128))
    
   # Convert image to numpy array and normalize
    img_array = np.array(img) / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict the class of the digit
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
   # Return the predicted class as a JSON response
    return {"predicted_class": class_names[predicted_class]}
