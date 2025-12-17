# Use official Python base image
FROM python:3.10-slim

# Allow statements and log messages to immediately appear
ENV PYTHONUNBUFFERED True

# Install git for CLIP
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set work dir
WORKDIR /app

# Copy dependencies
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /app

# Cloud Run provides PORT env variable
ENV PORT=8080

# Expose port (for local testing)
EXPOSE 8080

# Run FastAPI with uvicorn
CMD exec uvicorn app:app --host 0.0.0.0 --port $PORT
