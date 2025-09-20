# Miniconda environement 
FROM continuumio/miniconda3

# Get Mamba
RUN conda install -n base -c conda-forge mamba

# Set the working directory inside the container
WORKDIR /app

# Copy only the enviroment files to leverage Docker cache (so that packages are installed in lower layers)
# This means we can switch between cpu and gpu envs at build time
COPY environment*.yml .

# Create the environment (GPU version by default, can override with --build-arg ENV_FILE=environment_cpu.yml)
ARG ENV_FILE=environment_gpu.yml
ARG CUDA_VERSION=12.9
ENV CONDA_OVERRIDE_CUDA=${CUDA_VERSION}
RUN mamba env create -f ${ENV_FILE}

# Set env as default by putting it on PATH, and set the name of the env
ARG ENV_NAME=jfk-taxi-analysis-gpu
ENV PATH /opt/conda/envs/${ENV_NAME}/bin:$PATH

# For future reference this is how to mount a volume (allows data to persit accross containers)
# docker run -it --rm --mount source=my-volume,destination=/my-data/ ubuntu:22.04

# For development, mount the current directory to /app in the container 

# Build commands

# Builds the docker image with the tag jfk-taxi, uses defeault args
# docker build -t jfk-taxi . 

# GPU image
# docker build -t jfk-taxi:gpu --build-arg ENV_FILE=environment_gpu.yml --build-arg ENV_NAME=jfk-taxi-analysis-gpu .

# CPU image
# docker build -t jfk-taxi:cpu --build-arg ENV_FILE=environment_cpu.yml --build-arg ENV_NAME=jfk-taxi-analysis-cpu .    

# -t sets the tag (name) and the oppotional build args, . just means use this directory as the build context
