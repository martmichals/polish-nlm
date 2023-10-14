# Pytorch image w/CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Working directory, same name as the repository
WORKDIR /polish-nlm

# Install additional python dependencies
COPY requirements.txt /polish-nlm/
RUN pip install -r requirements.txt

# Install required packages
RUN apt-get update
RUN apt-get install -y curl unzip