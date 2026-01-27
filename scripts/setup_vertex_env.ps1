# Setup Google Cloud Vertex AI Environment for CTAFlow Training
# This script sets up the necessary GCP infrastructure for training CTAFlow models

# Configuration
$PROJECT_ID = Read-Host "Enter your GCP Project ID"
$REGION = "us-east4"
$REPO_NAME = "ctaflow-repo"
$BUCKET_NAME = Read-Host "Enter your GCS bucket name (must be globally unique)"

Write-Host "`n=== CTAFlow Vertex AI Setup ===" -ForegroundColor Cyan
Write-Host "Project ID: $PROJECT_ID" -ForegroundColor Yellow
Write-Host "Region: $REGION" -ForegroundColor Yellow
Write-Host "Repository: $REPO_NAME" -ForegroundColor Yellow
Write-Host "Bucket: $BUCKET_NAME`n" -ForegroundColor Yellow

# Step 1: Configure gcloud
Write-Host "[Step 1/5] Configuring gcloud..." -ForegroundColor Green
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION

# Step 2: Enable APIs
Write-Host "`n[Step 2/5] Enabling required APIs..." -ForegroundColor Green
gcloud services enable aiplatform.googleapis.com storage.googleapis.com artifactregistry.googleapis.com

# Step 3: Create GCS Bucket
Write-Host "`n[Step 3/5] Creating GCS bucket..." -ForegroundColor Green
$bucketExists = gsutil ls -b gs://$BUCKET_NAME 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Bucket gs://$BUCKET_NAME already exists. Skipping creation." -ForegroundColor Yellow
} else {
    gsutil mb -l $REGION gs://$BUCKET_NAME
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Bucket created successfully!" -ForegroundColor Green
    } else {
        Write-Host "Failed to create bucket. Check if name is available." -ForegroundColor Red
        exit 1
    }
}

# Step 4: Create Artifact Registry Repository
Write-Host "`n[Step 4/5] Creating Artifact Registry repository..." -ForegroundColor Green
$repoExists = gcloud artifacts repositories describe $REPO_NAME --location=$REGION 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Repository $REPO_NAME already exists. Skipping creation." -ForegroundColor Yellow
} else {
    gcloud artifacts repositories create $REPO_NAME `
        --repository-format=docker `
        --location=$REGION `
        --description="CTAFlow Docker repository"

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Repository created successfully!" -ForegroundColor Green
    } else {
        Write-Host "Failed to create repository." -ForegroundColor Red
        exit 1
    }
}

# Step 5: Configure Docker authentication
Write-Host "`n[Step 5/5] Configuring Docker authentication..." -ForegroundColor Green
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

# Save configuration to a file for use by job creation script
$configFile = "vertex_config.json"
$config = @{
    PROJECT_ID = $PROJECT_ID
    REGION = $REGION
    REPO_NAME = $REPO_NAME
    BUCKET_NAME = $BUCKET_NAME
    IMAGE_URI = "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/recurrent-wspr-trainer:latest"
} | ConvertTo-Json

$config | Out-File -FilePath $configFile -Encoding UTF8

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "Configuration saved to: $configFile" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Upload your training data to gs://$BUCKET_NAME/{ticker}/" -ForegroundColor White
Write-Host "2. Build and push Docker image using: .\build_and_push_image.ps1" -ForegroundColor White
Write-Host "3. Submit training jobs using: .\submit_training_job.ps1" -ForegroundColor White
