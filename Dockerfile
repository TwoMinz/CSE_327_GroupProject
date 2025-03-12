# Base image with Python support
FROM python:3.10-slim

# Working directory
WORKDIR /app

# System packages needed for OpenCV and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Set up directories
RUN mkdir -p /app/data /app/models /app/outputs
RUN chmod -R 777 /app

# Default command
CMD ["/bin/bash"]