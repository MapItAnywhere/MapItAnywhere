# Typical Usage:
# docker image build . -t mia:release
# docker run -v /home/:/home/ --network=bridge -it mia:release
# Add '--gpus all' for gpu support

# For CPU
# FROM ubuntu:20.04
# For GPU
FROM nvidia/cuda:12.2.2-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install apt-getable dependencies
RUN apt-get update \
    && apt-get install -y \
        build-essential \
        cmake \
        git \
        libeigen3-dev \
        libopencv-dev \
        libceres-dev \
        python3-dev \
        curl \
        pkg-config \
        libcairo2-dev \
        software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Mapmachine requirements
RUN add-apt-repository ppa:ubuntugis/ppa && \
    apt-get update && \
    apt-get -y install libgeos-dev

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.9-dev && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py

ARG REINSTALL_MAPMACHINE=1
RUN pip3.9 install git+https://github.com/tonyzzzzzz/map-machine

WORKDIR /home/

# OrienterNet Requirements TODO: Install directly from our requirements once our repo is public

RUN git clone https://github.com/mapillary/OpenSfM.git && cd OpenSfM && \
    pip3.9 install -r requirements.txt 

RUN git clone https://github.com/facebookresearch/OrienterNet.git && cd OrienterNet && \
    pip3 install -r requirements/full.txt

# MapPerceptionNet extra requirements
RUN pip3.9 install geojson shapely geopandas mercantile turfpy vt2geojson folium \
    geopy gradio_client pyarrow cloudpickle==2.0.0 urllib3~=1.25.6 scikit-image filelock hydra-core

# Earth Engine requirements (Required if sattelite image support is needed)

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && apt-get install google-cloud-sdk -y

RUN pip3.9 install earthengine-api

# Run these once you are in the docker with your credentials and google earth project
# earthengine authenticate
# gcloud auth application-default set-quota-project PROJECT_ID