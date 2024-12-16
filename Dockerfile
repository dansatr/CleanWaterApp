FROM python:3.8

WORKDIR /app

# Install system dependencies 
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN python -m pip install --upgrade pip && \
    python -m pip install wheel setuptools && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Environment variables
ENV PORT=5000

# Expose the port
EXPOSE $PORT

# Use gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT app:app