# Use Ollama as a base image
FROM ollama/base:latest

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5001

# Run the FastAPI app
CMD ["uvicorn", "app:fastapi_app", "--host", "0.0.0.0", "--port", "5001"]
