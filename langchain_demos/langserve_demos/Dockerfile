FROM python:3.11-slim


# Ensure pip is installed
RUN apt-get update && apt-get install -y python3-pip

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5001

# Run the FastAPI app
CMD ["python", "app.py"]

