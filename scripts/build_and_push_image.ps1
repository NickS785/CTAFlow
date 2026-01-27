# Build and Push CTAFlow Docker Image to Artifact Registry
# This script builds the training container and pushes it to GCP

# Load configuration
$configFile = "vertex_config.json"
if (-not (Test-Path $configFile)) {
    Write-Host "Error: Configuration file not found. Run setup_vertex_env.ps1 first." -ForegroundColor Red
    exit 1
}

$config = Get-Content $configFile | ConvertFrom-Json
$PROJECT_ID = $config.PROJECT_ID
$REGION = $config.REGION
$REPO_NAME = $config.REPO_NAME
$IMAGE_URI = $config.IMAGE_URI

Write-Host "`n=== Building and Pushing CTAFlow Docker Image ===" -ForegroundColor Cyan
Write-Host "Image URI: $IMAGE_URI`n" -ForegroundColor Yellow

# Check if Dockerfile exists
if (-not (Test-Path "Dockerfile")) {
    Write-Host "Error: Dockerfile not found. Run this script from the CTAFlow project root." -ForegroundColor Red
    exit 1
}

# Build the Docker image
Write-Host "[Step 1/2] Building Docker image..." -ForegroundColor Green
Write-Host "This may take 10-15 minutes due to TA-Lib compilation..." -ForegroundColor Yellow
docker build -t $IMAGE_URI -f Dockerfile .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build failed!" -ForegroundColor Red
    exit 1
}

# Push the image to Artifact Registry
Write-Host "`n[Step 2/2] Pushing image to Artifact Registry..." -ForegroundColor Green
docker push $IMAGE_URI

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker push failed! Make sure you're authenticated with 'gcloud auth configure-docker'." -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Image Build Complete ===" -ForegroundColor Cyan
Write-Host "Image successfully pushed to: $IMAGE_URI" -ForegroundColor Green
Write-Host "`nYou can now submit training jobs using: .\submit_training_job.ps1" -ForegroundColor Yellow
