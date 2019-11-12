# Use an official Python runtime as a parent image
FROM python:3.6-slim

# Install any needed packages specified in requirements.txt
RUN python -m pip install numpy scipy scikit-learn 

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
