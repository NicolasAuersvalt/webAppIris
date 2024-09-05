# Use a base image with Python
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the specific version of pip
RUN pip3 install --upgrade pip==24.2

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy your application code to the container
COPY . /app
WORKDIR /app

# Set the command to run your app
CMD ["streamlit", "run", "webApp.py"]
