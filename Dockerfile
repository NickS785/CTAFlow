# Use an official PyTorch image as a parent image.
# Using PyTorch 2.x with CUDA 11.8 for better compatibility.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory in the container.
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib

# Clone and install SierraPy (dependency for CTAFlow)
RUN git clone https://github.com/NickS785/SierraPy.git /app/SierraPy && \
    pip install --no-cache-dir -e /app/SierraPy

# Clone CTAFlow repository
RUN git clone https://github.com/NickS785/CTAFlow.git /app/CTAFlow

# Install CTAFlow requirements and package
RUN pip install --no-cache-dir -r /app/CTAFlow/requirements.txt && \
    pip install --no-cache-dir TA-Lib && \
    pip install --no-cache-dir -e /app/CTAFlow

# Environment variables for GCS bucket paths
# Data bucket: gs://fin_data_eod2/{ticker.lower()}
# Environment bucket: gs://ctaflow-env/
ENV DATA_GCS_BUCKET="gs://fin_data_eod2"
ENV ENV_GCS_BUCKET="gs://ctaflow-env"

# Set working directory to CTAFlow
WORKDIR /app/CTAFlow

# The training script will be the entrypoint.
# The arguments to the script will be passed by Vertex AI at runtime.
#
# Example usage with custom loss function:
#   --data-gcs-path gs://fin_data_eod2 \
#   --output-gcs-path gs://ctaflow-env/models \
#   --ticker cl \
#   --task classification \
#   --loss OrdinalCE \
#   --epochs 50
#
# Available loss functions: CrossEntropy, ExpectedPnL, OrdinalCE, Hierarchical, CostAwareCE
ENTRYPOINT ["python", "scripts/train_vertex.py"]
