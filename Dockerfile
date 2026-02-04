# Use PyTorch 2.3.0 with CUDA 12.1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set working directory to /workspace/app
# This ensures "notebooks/" and "scripts/" are visible in the file browser
WORKDIR /workspace/app

# Install system dependencies
# Added curl/unzip for AWS CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2 (for S3 access)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm awscliv2.zip

# Clone and install SierraPy dependency
RUN git clone https://github.com/NickS785/SierraPy.git /workspace/app/SierraPy && \
    pip install --no-cache-dir -e /workspace/app/SierraPy

# Copy requirements and install
COPY requirements.txt pyproject.toml ./
# Install CTAFlow requirements + Jupyter
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir boto3 jupyterlab ipywidgets

# Copy the project files
COPY . .

# Install CTAFlow package
RUN pip install --no-cache-dir -e .

# Environment Variables (S3 Buckets)
ENV DATA_S3_BUCKET="s3://fin_data_eod2"

# Expose the standard Jupyter port
EXPOSE 8888

# --- NEW ENTRYPOINT ---
# Launches JupyterLab on port 8888, accessible from outside
# Token is disabled for instant access (Be careful if sharing the URL)
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]