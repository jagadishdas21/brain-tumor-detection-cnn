from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from flask_cors import CORS
import os
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Enable logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Load the model, with a check for the existence of the model file
model_path = os.path.join(os.getcwd(), 'model.keras')
if os.path.exists(model_path):
    model = load_model(model_path)
    logging.info("Model loaded successfully")
else:
    logging.error(f"Model file not found at {model_path}")
    model = None  # Avoid attempting to use an undefined model later

def preprocess_image(image):
    try:
        img = image.resize((256, 256))  # Resize image to the input size your model expects
        img = np.array(img)
        img = img.reshape(1, 256, 256, 3) / 255.0  # Normalizing image
        return img
    except Exception as e:
        logging.error(f"Error in preprocessing image: {str(e)}")
        return None

# Serving the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Handling prediction request
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500  # Return if model loading failed

    try:
        logging.info("Prediction request received")

        # Check if a file was uploaded
        if 'file' not in request.files:
            logging.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        logging.info(f"Received file: {file.filename}")

        # Check if a file was selected
        if file.filename == '':
            logging.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        # Open and preprocess the image
        try:
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            logging.info("Image loaded successfully")
        except Exception as e:
            logging.error(f"Error opening image: {str(e)}")
            return jsonify({'error': 'Invalid image file'}), 400

        preprocessed_image = preprocess_image(image)
        if preprocessed_image is None:
            return jsonify({'error': 'Image preprocessing failed'}), 500

        logging.debug(f"Preprocessed image shape: {preprocessed_image.shape}")

        # Make prediction
        prediction = model.predict(preprocessed_image)
        logging.info(f"Raw prediction: {prediction}")

        # Get predicted class and confidence level
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

        logging.info(f"Predicted class: {predicted_class}, Confidence: {confidence}")

        return jsonify({'prediction': int(predicted_class), 'confidence': float(confidence)}), 200
    
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500  # Handle unexpected errors

if __name__ == '__main__':
    # Run the app on the local server at port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)