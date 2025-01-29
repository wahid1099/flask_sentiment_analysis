# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all necessary files, including the model and vectorizer
COPY . .  

# Expose Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
