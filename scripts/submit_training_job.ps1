# Submit CTAFlow Training Job to Vertex AI
# This script submits a custom training job to Vertex AI

param(
    [Parameter(Mandatory=$false)]
    [string]$Ticker,

    [Parameter(Mandatory=$false)]
    [string]$Task = "classification",

    [Parameter(Mandatory=$false)]
    [string]$Loss = "OrdinalCE",

    [Parameter(Mandatory=$false)]
    [int]$Epochs = 50,

    [Parameter(Mandatory=$false)]
    [int]$BatchSize = 32,

    [Parameter(Mandatory=$false)]
    [float]$LearningRate = 0.0001,

    [Parameter(Mandatory=$false)]
    [int]$WindowDays = 10,

    [Parameter(Mandatory=$false)]
    [string]$MachineType = "n1-standard-8",

    [Parameter(Mandatory=$false)]
    [switch]$UseGPU,

    [Parameter(Mandatory=$false)]
    [string]$DataBucket,

    [Parameter(Mandatory=$false)]
    [string]$OutputBucket
)

# Load configuration
$configFile = "vertex_config.json"
if (-not (Test-Path $configFile)) {
    Write-Host "Error: Configuration file not found. Run setup_vertex_env.ps1 first." -ForegroundColor Red
    exit 1
}

$config = Get-Content $configFile | ConvertFrom-Json
$PROJECT_ID = $config.PROJECT_ID
$REGION = $config.REGION
$IMAGE_URI = $config.IMAGE_URI

# Use config bucket if not provided
if (-not $OutputBucket) {
    $OutputBucket = "gs://$($config.BUCKET_NAME)/models"
}

# Prompt for required parameters if not provided
if (-not $Ticker) {
    $Ticker = Read-Host "Enter ticker symbol (e.g., CL, NG, ZC)"
}

if (-not $DataBucket) {
    $DataBucket = Read-Host "Enter data GCS path (e.g., gs://fin_data_eod2)"
}

$Ticker = $Ticker.ToLower()

Write-Host "`n=== Submitting CTAFlow Training Job ===" -ForegroundColor Cyan
Write-Host "Ticker: $Ticker" -ForegroundColor Yellow
Write-Host "Task: $Task" -ForegroundColor Yellow
Write-Host "Loss: $Loss" -ForegroundColor Yellow
Write-Host "Epochs: $Epochs" -ForegroundColor Yellow
Write-Host "Batch Size: $BatchSize" -ForegroundColor Yellow
Write-Host "Learning Rate: $LearningRate" -ForegroundColor Yellow
Write-Host "Window Days: $WindowDays" -ForegroundColor Yellow
Write-Host "Machine Type: $MachineType" -ForegroundColor Yellow
Write-Host "Use GPU: $UseGPU" -ForegroundColor Yellow
Write-Host "Data Bucket: $DataBucket" -ForegroundColor Yellow
Write-Host "Output Bucket: $OutputBucket`n" -ForegroundColor Yellow

# Generate job name with timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$JOB_NAME = "ctaflow_wspr_${Ticker}_${timestamp}"

# Build arguments for the training script
$args_list = @(
    "--data-gcs-path=$DataBucket",
    "--output-gcs-path=$OutputBucket",
    "--ticker=$Ticker",
    "--task=$Task",
    "--loss=$Loss",
    "--epochs=$Epochs",
    "--batch-size=$BatchSize",
    "--learning-rate=$LearningRate",
    "--window-days=$WindowDays"
)

# Join arguments for gcloud command
$args_string = ($args_list | ForEach-Object { """$_""" }) -join ","

# Build worker pool spec
$worker_spec = "machine-type=$MachineType,replica-count=1,container-image-uri=$IMAGE_URI"

# Add GPU if requested
if ($UseGPU) {
    $worker_spec += ",accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1"
}

Write-Host "Submitting job: $JOB_NAME" -ForegroundColor Green

# Submit the job
$cmd = "gcloud ai custom-jobs create " +
       "--region=$REGION " +
       "--display-name=$JOB_NAME " +
       "--worker-pool-spec=""$worker_spec"" " +
       "--args=$args_string"

Invoke-Expression $cmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== Job Submitted Successfully ===" -ForegroundColor Cyan
    Write-Host "Job Name: $JOB_NAME" -ForegroundColor Green
    Write-Host "`nMonitor job at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID" -ForegroundColor Yellow
    Write-Host "`nView logs with: gcloud ai custom-jobs stream-logs [JOB_ID] --region=$REGION" -ForegroundColor White
    Write-Host "`nTrained model will be saved to: $OutputBucket/$Ticker/best_model.pt" -ForegroundColor White
} else {
    Write-Host "`nJob submission failed!" -ForegroundColor Red
    exit 1
}
