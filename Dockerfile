FROM python:3.8-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Environment variables
ENV PORT=5000

# Expose the port
EXPOSE $PORT

# Use gunicorn instead of Flask's development server
CMD gunicorn --bind 0.0.0.0:$PORT app:app