FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    vim \
    ffmpeg \
    libsm6 \
    libxext6

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/mapper

COPY --chown=user . $HOME/mapper

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Get Weights
RUN bash get_weights.sh