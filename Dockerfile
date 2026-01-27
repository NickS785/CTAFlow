# Use an official PyTorch image as a parent image.
# Using PyTorch 2.x with CUDA 11.8 for better compatibility.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory in the container.
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone and install SierraPy (dependency for CTAFlow)
RUN git clone https://github.com/NickS785/SierraPy.git /app/SierraPy && \
    pip install --no-cache-dir -e /app/SierraPy

# Copy requirements and pyproject.toml first for better layer caching
COPY requirements.txt pyproject.toml ./
COPY CTAFlow/__init__.py CTAFlow/__init__.py

# Install CTAFlow requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the CTAFlow project
# .dockerignore will filter out unnecessary files
COPY . .

# Install CTAFlow package
RUN pip install --no-cache-dir -e .

# Environment variables for GCS bucket paths
# Data bucket: gs://fin_data_eod2/{ticker.lower()}
# Environment bucket: gs://ctaflow-env/
ENV DATA_GCS_BUCKET="gs://fin_data_eod2"
ENV ENV_GCS_BUCKET="gs://ctaflow-env"

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
