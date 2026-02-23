# CTAFlow Training Scripts

This directory contains scripts for training CTAFlow models both locally and on Google Cloud Vertex AI.

## Vertex AI Training Scripts (PowerShell)

Automated PowerShell scripts for Windows users to deploy training jobs to Google Cloud Vertex AI.

### Quick Start

1. **Setup GCP Environment** (One-time setup)
   ```powershell
   .\setup_vertex_env.ps1
   ```
   - Configures gcloud with region `us-east4`
   - Enables required APIs
   - Creates GCS bucket and Artifact Registry repository `ctaflow-repo`
   - Saves configuration to `vertex_config.json`

2. **Build and Push Docker Image** (After code changes)
   ```powershell
   .\build_and_push_image.ps1
   ```
   - Builds Docker container with CTAFlow and dependencies
   - Pushes to Artifact Registry (takes 10-15 minutes)

3. **Submit Training Job**
   ```powershell
   # Basic usage
   .\submit_training_job.ps1 -Ticker "cl" -DataBucket "gs://fin_data_eod2"

   # Advanced usage with custom loss and GPU
   .\submit_training_job.ps1 `
       -Ticker "ng" `
       -DataBucket "gs://fin_data_eod2" `
       -Loss "OrdinalCE" `
       -Epochs 100 `
       -BatchSize 64 `
       -UseGPU
   ```

### submit_training_job.ps1 Parameters

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `-Ticker` | Ticker symbol (lowercase) | - | Yes |
| `-DataBucket` | GCS path to data | - | Yes |
| `-Task` | "classification" or "regression" | "classification" | No |
| `-Loss` | Loss function (see below) | "OrdinalCE" | No |
| `-Epochs` | Number of epochs | 50 | No |
| `-BatchSize` | Batch size | 32 | No |
| `-LearningRate` | Learning rate | 0.0001 | No |
| `-WindowDays` | Rolling window size | 10 | No |
| `-MachineType` | GCP machine type | "n1-standard-8" | No |
| `-UseGPU` | Enable GPU (T4) | False | No |
| `-OutputBucket` | GCS output path | (from config) | No |

### Available Loss Functions

For classification tasks (`-Task "classification"`):

- **CrossEntropy**: Standard cross-entropy loss
- **ExpectedPnL**: Direct P&L optimization (requires raw returns)
- **OrdinalCE**: Ordinal CE with anti-collapse regularization (penalizes opposite direction mistakes)
- **Hierarchical**: Two-stage hierarchical loss (direction → full class)
- **CostAwareCE**: Transaction cost-aware cross-entropy

Example:
```powershell
# Use ordinal loss with anti-collapse
.\submit_training_job.ps1 -Ticker "cl" -Loss "OrdinalCE" -Epochs 100

# Optimize expected P&L directly
.\submit_training_job.ps1 -Ticker "ng" -Loss "ExpectedPnL" -Epochs 75
```

## Python Training Scripts

### train_vertex.py

Main training script for Vertex AI. Supports:
- Multi-modal deep learning (RecurrentWSPR model)
- Custom loss functions for trading
- GCS data loading and model saving
- GPU acceleration

Called automatically by Vertex AI container. Arguments:
- `--data-gcs-path`: GCS bucket path (e.g., `gs://fin_data_eod2`)
- `--output-gcs-path`: GCS output path for models
- `--ticker`: Ticker symbol (lowercase)
- `--task`: "classification" or "regression"
- `--loss`: Loss function name
- `--epochs`, `--batch-size`, `--learning-rate`, etc.

## File Organization

```
scripts/
├── README.md                      # This file
├── setup_vertex_env.ps1          # GCP environment setup
├── build_and_push_image.ps1      # Docker build and push
├── submit_training_job.ps1       # Submit Vertex AI training job
├── train_vertex.py               # Python training script
└── vertex_config.json            # Generated config (gitignored)
```

## Data Organization

Training data should be organized in GCS as:
```
gs://your-bucket/{ticker}/
├── intraday.csv
├── features.csv
├── target.csv
├── vpin.parquet
├── profiles.npz
└── rasterized.npz
```

For example, with `gs://fin_data_eod2`:
```
gs://fin_data_eod2/cl/
gs://fin_data_eod2/ng/
gs://fin_data_eod2/zc/
```

## Monitoring Jobs

View job status in GCP Console:
```
https://console.cloud.google.com/vertex-ai/training/custom-jobs
```

Stream logs from command line:
```powershell
gcloud ai custom-jobs list --region=us-east4
gcloud ai custom-jobs stream-logs [JOB_ID] --region=us-east4
```

## Retrieving Trained Models

Models are saved to `{output-bucket}/{ticker}/best_model.pt`.

Download with:
```powershell
gsutil cp gs://your-bucket/models/cl/best_model.pt .
```

## Troubleshooting

### "Configuration file not found"
Run `.\setup_vertex_env.ps1` first to generate `vertex_config.json`.

### "Docker build failed"
- Ensure Docker Desktop is running
- Check disk space (build requires ~5GB)
- TA-Lib compilation may take 10-15 minutes

### "Docker push failed"
Run authentication:
```powershell
gcloud auth configure-docker us-east4-docker.pkg.dev
```

### "Job submission failed"
- Verify image was pushed successfully
- Check that data exists at specified GCS path
- Ensure GCS bucket names are correct

## See Also

- [VERTEX_AI_TRAINING_GUIDE.md](../VERTEX_AI_TRAINING_GUIDE.md) - Complete setup guide
- [Dockerfile](../Dockerfile) - Container definition
- [CTAFlow/models/deep_learning/loss/](../CTAFlow/models/deep_learning/training/loss/) - Loss function implementations
