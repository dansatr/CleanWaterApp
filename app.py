from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
import numpy as np
import cv2
from PIL import Image
import io
import os

app = Flask(__name__)

model = None

def load_model():
    global model
    try:
        # Create model with architecture
        input_shape = (224, 224, 3)
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        base_model.trainable = False
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        
        # Load weights from h5 file
        model.load_weights('best_water_quality_model_fold2.h5')
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("Successfully loaded model")
        return
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        
    print("Failed to load model")
    

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
