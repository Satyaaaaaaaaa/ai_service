# Force Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything into container
COPY . .

# System dependencies for TensorFlow CPU
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose server port
EXPOSE 5001

# Start Flask app via Gunicorn
CMD ["bash", "-c", "gunicorn app:app --timeout 600 --preload -b 0.0.0.0:$PORT"]

