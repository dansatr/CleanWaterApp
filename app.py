from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os

app = Flask(__name__)

model = None

# Load the model
def load_model():
    global model
    model_files = [
        'best_water_quality_model_fold1.h5',
        'best_water_quality_model_fold2.h5'
    ]
    
    for model_file in model_files :
        try:
            print(f"Attempting to load model from {model_file}")
            if os.path.exists(model_file):
                # Load model without compiling
                model = tf.keras.models.load_model(model_file, compile=False)
                
                # Recompile the model
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                print(f"Successfully loaded model from {model_file}")
                return
            else:
                print(f"Model file not found: {os.path.abspath(model_file)}")
        except Exception as e:
            print(f"Error loading {model_file}: {str(e)}")
    
    print("Failed to load any model file")
    

# Load model when starting the app
load_model()

def preprocess_image(image_bytes):
    # Convert bytes to image
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    
    # Convert to np array and resize
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    
    # Apply preprocessing steps from training
    image = cv2.GaussianBlur(image, (3,3), 0)
    
    # Apply CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    return np.expand_dims(image, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please ensure model file exists and is valid.'})
        
    try:
        # Get image from request
        file = request.files['file']
        image_bytes = file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        prediction = model.predict(processed_image)[0][0]
        
        # Convert prediction to class label
        label = "Clean" if prediction < 0.5 else "Dirty"
        confidence = float(prediction if prediction >= 0.5 else 1 - prediction)
        
        return jsonify({
            'prediction': label,
            'confidence': f"{confidence:.2%}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
