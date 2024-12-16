# Water Quality Classification System

A deep learning-based system for classifying water quality using image analysis.

## Project Overview
This project implements a water quality classification system using deep learning. It can classify water images as clean or dirty using a fine-tuned MobileNetV2 model.

## Features
- Web interface for image upload and classification
- Real-time prediction with confidence scores
- Responsive design with instant feedback
- Pre-trained model using transfer learning

## Technical Architecture
- Backend: Flask (Python)
- Frontend: HTML, CSS, JavaScript
- Model: MobileNetV2 (Transfer Learning)
- Training: K-fold Cross-validation

## Model Performance
- Accuracy: 86.90%
- Best F1 Score: 0.9333
- Cross-validation with 3 folds

├── app.py                      # Flask application
├── templates/                  # Frontend templates
│   └── index.html             # Main web interface
├── test.ipynb                 # Model training notebook
├── training_history_*.json    # Training logs
└── best_water_quality_model_*.h5  # Trained models

Depenedencies


Dependencies

Python 3.8+
TensorFlow 2.x
Flask
OpenCV
NumPy
Pillow
