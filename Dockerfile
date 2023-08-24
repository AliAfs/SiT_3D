# Use a base image compatible with the target system (x86_64 for CUDA)
FROM nvidia/cuda:11.0-base

# Set up the environment
RUN apt-get update && \
    apt-get install -y python3 python3-pip

# Install CUDA-related dependencies
# Install cuDNN, libraries, and other dependencies as needed
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

# Copy your repository into the container
COPY . /app
WORKDIR /app

# Set environment variables if needed

# Run your script or application
CMD ["python3", "main_test.py", "--save_recon", "--data_location", "path/to/data", "--batch_size", "?"]
