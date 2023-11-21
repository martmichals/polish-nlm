# Pytorch image w/CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install required packages
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y curl unzip zip git

# Install additional python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
