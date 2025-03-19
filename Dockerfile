# Python runtime as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /src

COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

# Expose the port your Flask app will run on
EXPOSE 5000

# Command to run your application
CMD ["python", "app.py"]
