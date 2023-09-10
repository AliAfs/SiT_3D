# Use a runtime image compatible with the target system
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Set up the environment
RUN apt-get update && \
    apt-get install -y python3 python3-pip

# Install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

# Copy your repository into the container
#COPY . /app
WORKDIR /app

# Set environment variables if needed

# Run your script or application
#CMD ["python3", "main_3D.py"]
