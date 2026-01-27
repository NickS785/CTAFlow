# Guide to Training CTAFlow Models on Google Cloud Vertex AI

This guide provides step-by-step instructions for training a custom `CTAFlow` deep learning model on Google Cloud's Vertex AI platform. We will use a custom container approach, which gives us full control over the training environment.

This guide assumes you are training a `RecurrentWSPR` model using the provided `scripts/train_vertex.py` script.

## Prerequisites

1.  **Google Cloud Account**: You need a Google Cloud account with billing enabled.
2.  **Google Cloud SDK**: Install and initialize the `gcloud` CLI.
3.  **Docker**: Install Docker on your local machine.

## Step 1: Setup Your Google Cloud Environment

### 1.1 Configure gcloud

Set your project ID and region.

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1" # Or any other supported region

gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION
```

### 1.2 Enable APIs

Enable the necessary APIs for Vertex AI, Cloud Storage, and Artifact Registry.

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com
```

### 1.3 Create a GCS Bucket

Create a Google Cloud Storage bucket to store your data and trained models.

```bash
export BUCKET_NAME="your-unique-bucket-name"
gsutil mb -l $REGION gs://$BUCKET_NAME
```

### 1.4 Create an Artifact Registry Repository

Create a Docker repository in Artifact Registry to store your training container image.

```bash
export REPO_NAME="ctaflow-repo"
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="CTAFlow Docker repository"
```

## Step 2: Prepare Your Data

The `scripts/train_vertex.py` script expects several data files organized by ticker. You need to upload these to your GCS bucket under a directory named after the ticker.

**Required files for each ticker:**

-   Intraday data (e.g., `intraday.csv`)
-   Summary features (e.g., `features.csv`)
-   Target values (e.g., `target.csv`)
-   VPIN data (e.g., `vpin.parquet`)
-   Profile data (e.g., `profiles.npz`)
-   Rasterized VPIN data (e.g., `rasterized.npz`)

Create a `data/{TICKER}` directory in your GCS bucket and upload the files. For example, for ticker `CL`:

```bash
export TICKER="CL"
gsutil cp path/to/your/local/intraday.csv gs://$BUCKET_NAME/data/${TICKER}/intraday.csv
gsutil cp path/to/your/local/features.csv gs://$BUCKET_NAME/data/${TICKER}/features.csv
gsutil cp path/to/your/local/target.csv gs://$BUCKET_NAME/data/${TICKER}/target.csv
gsutil cp path/to/your/local/vpin.parquet gs://$BUCKET_NAME/data/${TICKER}/vpin.parquet
gsutil cp path/to/your/local/profiles.npz gs://$BUCKET_NAME/data/${TICKER}/profiles.npz
gsutil cp path/to/your/local/rasterized.npz gs://$BUCKET_NAME/data/${TICKER}/rasterized.npz
```

Your data for ticker `CL` will be at `gs://your-unique-bucket-name/data/CL/`.

## Step 3: Package the Training Application

We will use the provided `Dockerfile` to build a container image for our training application.

### 3.1 Define Image URI

```bash
export IMAGE_NAME="recurrent-wspr-trainer"
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
```

### 3.2 Build and Push the Docker Image

From the root of the `CTAFlow` project, run the following commands:

```bash
# Configure Docker to use gcloud as a credential helper
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build the Docker image
docker build -t $IMAGE_URI -f Dockerfile .

# Push the image to Artifact Registry
docker push $IMAGE_URI
```

## Step 4: Launch the Vertex AI Training Job

Now, we submit the custom training job to Vertex AI.

```bash
# Define job details
export TICKER="CL" # The ticker you want to train on
export JOB_NAME="ctaflow_wspr_training_${TICKER}_$(date +%Y%m%d_%H%M%S)"
export DATA_GCS_PATH="gs://${BUCKET_NAME}/"
export OUTPUT_GCS_PATH="gs://${BUCKET_NAME}/models"

# Arguments for the training script
# These are passed to scripts/train_vertex.py
TRAINER_ARGS="--data-gcs-path=${DATA_GCS_PATH},"
TRAINER_ARGS+="--output-gcs-path=${OUTPUT_GCS_PATH},"
TRAINER_ARGS+="--ticker=${TICKER},"
TRAINER_ARGS+="--intraday-filename=intraday.csv,"
TRAINER_ARGS+="--features-filename=features.csv,"
TRAINER_ARGS+="--target-filename=target.csv,"
TRAINER_ARGS+="--vpin-filename=vpin.parquet,"
TRAINER_ARGS+="--profile-filename=profiles.npz,"
TRAINER_ARGS+="--rasterized-filename=rasterized.npz,"
TRAINER_ARGS+="--epochs=50,"
TRAINER_ARGS+="--batch-size=32,"
TRAINER_ARGS+="--learning-rate=1e-4"

# Submit the job
gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=$JOB_NAME \
  --worker-pool-spec="machine-type=n1-standard-4,replica-count=1,executor-image-uri=${IMAGE_URI},args=[${TRAINER_ARGS}]"
```

If you have a GPU-enabled machine type (e.g., `n1-standard-4` with `accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1`), Vertex AI will use the CUDA-enabled Docker image.

## Step 5: Monitor the Job and Retrieve the Model

### 5.1 Monitor the Job

You can monitor the job's progress and view logs in the Google Cloud Console under `Vertex AI > Training > Custom Jobs`.

To view logs from the command line:

```bash
# Find your job ID first in the console or by listing jobs:
gcloud ai custom-jobs list --region=$REGION

# Then stream logs
gcloud ai custom-jobs stream-logs [JOB_ID] --region=$REGION
```

### 5.2 Retrieve the Model

Once the job is complete, the trained model (`best_model.pt`) will be saved in your GCS bucket in a ticker-specific directory: `gs://your-unique-bucket-name/models/{TICKER}/best_model.pt`.

You can download it using `gsutil`:

```bash
gsutil cp gs://$BUCKET_NAME/models/${TICKER}/best_model.pt .
```

## Next Steps

After training, you can:
-   **Deploy the model**: Deploy the trained model to a Vertex AI Endpoint for real-time predictions.
-   **Hyperparameter Tuning**: Use Vertex AI Hyperparameter Tuning to find the best hyperparameters for your model.
-   **Batch Predictions**: Use the model for batch predictions on new data.
