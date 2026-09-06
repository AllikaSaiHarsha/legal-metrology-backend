FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for OpenCV and EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create an uploads directory
RUN mkdir -p /app/uploads && chmod 777 /app/uploads

# Copy application code
COPY . .

# Set permissions for HF Spaces
RUN chmod -R 777 /app

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Run FastAPI on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
