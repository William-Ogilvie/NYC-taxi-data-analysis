# Micro mamba environement, we want a CUDA base image for GPU support 
FROM mambaorg/micromamba:cuda12.9.1-ubuntu22.04
# handy command to check cuda version on machine -nvidia-smi

# Using this guide
# https://micromamba-docker.readthedocs.io/en/latest/quick_start.html 

# Choose the env to run, cpu or gpu
ARG ENV_FILE=environment_gpu.yml

# chown sets the owner of the copied files to the mamba user not the root user
COPY --chown=$MAMBA_USER:$MAMBA_USER $ENV_FILE /tmp/$ENV_FILE

# Create the environment and clean up, note we have to set the name to be base (well we could have mutliptle envs but we don't need to for this project)
RUN micromamba install -y -n base -f /tmp/$ENV_FILE && \
    micromamba clean --all --yes

# Set as working directory
WORKDIR /app

# To build the docker image run (select the env you want to use with --build-arg ENV_FILE=environment_cpu.yml or environment_gpu.yml)
# docker build -t jfk-taxi:gpu --build-arg ENV_FILE=environment_gpu.yml .

# The base conda env will be automatically activated when we run the image:
# docker run -it --rm jfk-taxi:gpu python --version (should display python version)
# -i interactive mode (keep STDIN open)
# -t allocate a pseudo-TTY (formatting of terminal)
# --rm auto remove the container when it exits

# If you just want to run the container remove the python --version part
# docker run -it --rm jfk-taxi:gpu

# We want to create a bind mount of current directory to /app in the container so we can access our code and data
# the reason for a bind mount is that it allows us to edit files in the host system, i.e. we can save results from modelling into the data/saved_objects dir
# This means we can use the container as a dev environment, changes here exist in the container and vice versa
# docker run -it --rm --mount type=bind,src="$(pwd)",dst=/app jfk-taxi:gpu


# When we run the container we will need to install our local pacakge jfk_taxis, to do this write pip install -e . 
# In the EC2 instance we run into problems with not having permissions inside the container to create the jfk_taxis.egg-info directory
# to fix this we can ensure the container runs with the same UID (user ID) and GID (group ID) as the host user, meaning we have the same 
# read write permsissions in the container as the host user on the files passed in the bind mount

# To do this we can use the following command to get the UID and GID of the host user:
# -u $(id -u):$(id -g)
# docker run -it --rm -u $(id -u):$(id -g) --mount type=bind,src="$(pwd)",dst=/app jfk-taxi:gpu

# To run jupyter lab we use the following command

# We also want to map port the host port 8889 to the container port 8888 so we can access jupyter lab
# and also map port 8501 for streamlit
# docker run -it --rm -p 8888:8888 -p 8501:8501 --mount type=bind,src="$(pwd)",dst=/app jfk-taxi:gpu 

# To actually run jupyter lab use the following command:
# jupyter lab --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token=''
# --ip=0.0.0.0 means listen on all interfaces, so host can reach it 
# --no-browser means don't try to open a browser in the container
# --allow-root means allow to run as root user 
# --NotebookApp.token='' means disable the token authentication (not secure but ok for local dev)

# We also need to expose the gpu to the container, this is done with --gpus all
# docker run -it --rm --gpus all -p 8888:8888 -p 8501:8501 --mount type=bind,src="$(pwd)",dst=/app jfk-taxi:gpu




# Old docker file below, kept for reference only 
# The base conda 
# Get Mamba
#RUN conda install -n base -c conda-forge mamba

# Set the working directory inside the container
#WORKDIR /app

# Copy only the enviroment files to leverage Docker cache (so that packages are installed in lower layers)
# This means we can switch between cpu and gpu envs at build time
#COPY environment*.yml .

# Create the environment (GPU version by default, can override with --build-arg ENV_FILE=environment_cpu.yml)
#ARG ENV_FILE=environment_gpu.yml
#ARG CUDA_VERSION=12.9
#ENV CONDA_OVERRIDE_CUDA=${CUDA_VERSION}
#RUN mamba env create -f ${ENV_FILE} && \
#    conda clean -afy && \ 
#    mamba clean --all -y && \
#    rm -rf /root/.cache/pip
# We also clean mamaba and conda caches to reduce image size

# Set env as default by putting it on PATH, and set the name of the env
#ARG ENV_NAME=jfk-taxi-analysis-gpu
#ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:$PATH

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
