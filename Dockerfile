FROM python:3.10-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Copy only requirements first (caching)
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your entire project
COPY . .

# Expose port (Railway will map automatically)
EXPOSE 8000

# Start your ML API using Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--timeout", "600", "--workers", "1"]
