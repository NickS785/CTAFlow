## CTAFlow Storage System

Production-ready storage infrastructure for model lifecycle management and data operations.

### Features

- **Multi-Backend Storage**: Seamless switching between local filesystem and AWS S3
- **Model Management**: Versioning, checkpointing, and metadata tracking
- **Operating Modes**: Training, Backtest, and Production modes with mode-specific validation
- **AWS Integration**: S3 support with custom endpoints, automatic caching, and bulk operations
- **Pluggable Architecture**: Easy to extend with custom storage backends

### Quick Start

#### Local Storage

```python
from CTAFlow.data.storage import ModelManager, ModelMode, LocalStorage

# Create storage backend
storage = LocalStorage('data/')

# Create model manager for training
manager = ModelManager(storage, mode=ModelMode.TRAINING)

# Save a trained model
manager.save_model(
    model=my_trained_model,
    model_name='wspr_HE_LE',
    metadata={
        'tickers': ['HE', 'LE'],
        'config': {'d_model': 128, 'window_days': 10},
    },
    metrics={'accuracy': 0.65, 'sharpe': 1.8},
)

# Load for inference
model, metadata = manager.load_model('wspr_HE_LE')
```

#### S3 Storage

```python
from CTAFlow.data.storage import (
    ModelManager,
    ModelMode,
    AWSClient,
    S3Storage,
    S3Config,
)

# Configure S3
s3_config = S3Config(
    bucket_name='ctaflow-production',
    region='us-east-1',
    # Optional: custom endpoint
    # endpoint_url='https://s3.custom.com',
)

# Create AWS client
aws_client = AWSClient(s3_config)

# Create S3 storage backend
storage = S3Storage(
    bucket_name='ctaflow-production',
    aws_client=aws_client,
    prefix='models/',  # Optional prefix for all paths
)

# Create model manager for production
manager = ModelManager(storage, mode=ModelMode.PRODUCTION)

# Load production model
model, metadata = manager.load_model('wspr_HE_LE')
```

#### Convenience Function

```python
from CTAFlow.data.storage import create_model_manager

# Local training
manager = create_model_manager(
    mode='training',
    local_dir='./models',
)

# S3 production
manager = create_model_manager(
    mode='production',
    use_s3=True,
    s3_config={
        'bucket_name': 'ctaflow-prod',
        'region': 'us-west-2',
    },
)
```

### AWS Client Features

#### Download Ticker Data

```python
from CTAFlow.data.storage import AWSClient, S3Config

# Initialize client
aws_client = AWSClient(S3Config(bucket_name='my-bucket'))

# Download all data files for a ticker
data_files = aws_client.download_ticker_data(
    ticker='HE',
    local_dir='data/HE',
)

# Returns dict: {'features': Path, 'profiles': Path, ...}
```

#### List Available Tickers

```python
# List all tickers in S3
tickers = aws_client.list_tickers(data_prefix='data/')
# ['HE', 'LE', 'CL', 'RB', ...]
```

#### Custom Operations

```python
# Upload file
aws_client.upload_file('local/model.pth', 'models/my_model.pth')

# Download file with caching
aws_client.download_file('models/my_model.pth', 'local/model.pth', use_cache=True)

# List objects
keys = aws_client.list_objects(prefix='models/', suffix='.pth')

# Check existence
exists = aws_client.object_exists('models/my_model.pth')
```

### ModelManager API

#### Save Model

```python
manager.save_model(
    model=model,                    # PyTorch model
    model_name='wspr_HE_LE',        # Identifier
    metadata=dict,                  # Model metadata
    optimizer=optimizer,            # Optional: for resuming training
    epoch=50,                       # Optional: current epoch
    metrics=dict,                   # Optional: training metrics
    overwrite=False,                # Prevent accidental overwrite
)
```

#### Load Model

```python
# Load with model class (for instantiation)
from CTAFlow.models.deep_learning.multi_branch.dual_model import MultiAssetWSPR

model, metadata = manager.load_model(
    model_name='wspr_HE_LE',
    model_class=MultiAssetWSPR,  # Optional: auto-instantiate
    device='cuda',
)

# Load checkpoint for resuming training
metadata = manager.load_checkpoint(
    model_name='wspr_HE_LE',
    model=existing_model,
    optimizer=existing_optimizer,
)
```

#### List Models

```python
models = manager.list_models()
# [
#   {
#     'model_name': 'wspr_HE_LE',
#     'path': 'models/wspr_HE_LE.pth',
#     'size': 12345678,
#     'modified': '2024-01-15T10:30:00',
#     'metadata': {...},
#   },
#   ...
# ]
```

#### Delete Model

```python
success = manager.delete_model('wspr_HE_LE')
```

### Operating Modes

#### Training Mode
- Save checkpoints with metrics
- Resume training from checkpoints
- Track training history

```python
manager = ModelManager(storage, mode=ModelMode.TRAINING)

# During training
manager.save_model(
    model=model,
    model_name='wspr_HE_LE',
    optimizer=optimizer,
    epoch=current_epoch,
    metrics={'train_loss': 0.15, 'val_loss': 0.18},
)

# Resume training
metadata = manager.load_checkpoint(
    model_name='wspr_HE_LE',
    model=model,
    optimizer=optimizer,
)
start_epoch = metadata['epoch'] + 1
```

#### Backtest Mode
- Load trained models
- Run historical simulations
- No model saving (read-only for models)

```python
manager = ModelManager(storage, mode=ModelMode.BACKTEST)

# Load model for backtesting
model, metadata = manager.load_model('wspr_HE_LE')

# Run backtest...
```

#### Production Mode
- Load production models
- Strict validation
- Read-only for models

```python
manager = ModelManager(storage, mode=ModelMode.PRODUCTION)

# Load production model
model, metadata = manager.load_model('wspr_HE_LE')

# Generate predictions...
```

### Storage Backends

#### LocalStorage

Filesystem-based storage for development and local testing.

```python
from CTAFlow.data.storage import LocalStorage

storage = LocalStorage(root_dir='./data')

# Upload (copy) file
storage.upload_file('model.pth', 'models/my_model.pth')

# Download (copy) file
storage.download_file('models/my_model.pth', 'local_copy.pth')

# List files
files = storage.list_files(prefix='models/', suffix='.pth')

# Check existence
exists = storage.exists('models/my_model.pth')

# Get metadata
metadata = storage.get_metadata('models/my_model.pth')
```

#### S3Storage

Cloud storage backend for production deployment.

```python
from CTAFlow.data.storage import S3Storage, AWSClient, S3Config

# Configure AWS
aws_client = AWSClient(S3Config(
    bucket_name='my-bucket',
    region='us-east-1',
))

storage = S3Storage(
    bucket_name='my-bucket',
    aws_client=aws_client,
    prefix='ctaflow/',  # Optional: all paths prefixed
)

# Same API as LocalStorage
storage.upload_file('model.pth', 'models/my_model.pth')
# Uploads to: s3://my-bucket/ctaflow/models/my_model.pth
```

#### Custom Backend

Implement `StorageBackend` interface for custom storage:

```python
from CTAFlow.data.storage import StorageBackend

class CustomStorage(StorageBackend):
    def exists(self, path: str) -> bool:
        # Implementation
        pass

    def upload_file(self, local_path, remote_path) -> str:
        # Implementation
        pass

    def download_file(self, remote_path, local_path):
        # Implementation
        pass

    def list_files(self, prefix='', suffix=''):
        # Implementation
        pass

    def delete_file(self, path) -> bool:
        # Implementation
        pass

    def get_metadata(self, path):
        # Implementation
        pass
```

### Environment Variables

AWS credentials can be provided via environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

### Expected S3 Structure

```
s3://bucket-name/
├── models/
│   ├── wspr_HE_LE.pth
│   ├── wspr_HE_LE_metadata.json
│   ├── wspr_CL_RB.pth
│   └── wspr_CL_RB_metadata.json
│
└── data/
    ├── HE/
    │   ├── features.csv
    │   ├── profiles.npz
    │   ├── vpin.parquet
    │   ├── rasterized.npz
    │   ├── target.csv
    │   └── intraday.csv
    ├── LE/
    │   └── ...
    └── {TICKER}/
        └── ...
```

### Best Practices

1. **Use Modes Appropriately**:
   - Training: Save checkpoints regularly
   - Backtest: Read-only model access
   - Production: Strict validation, no model updates

2. **Leverage Caching**:
   - AWSClient caches downloads automatically
   - Reduces S3 API calls and costs

3. **Version Control**:
   - Use descriptive model names: `wspr_HE_LE_v2_20240115`
   - Include metadata for reproducibility

4. **Metadata Management**:
   - Save configuration, hyperparameters, and metrics
   - Enables model comparison and debugging

5. **Security**:
   - Use environment variables for credentials
   - Never commit credentials to version control
   - Use IAM roles for production deployments

### Integration with CTAFlow

The storage system integrates with existing CTAFlow components:

```python
from CTAFlow.models.multi_asset import MultiAssetMomentum
from CTAFlow.models.deep_learning.multi_branch.dual_model import MultiAssetWSPR
from CTAFlow.data.storage import create_model_manager

# Train model
multi_asset = MultiAssetMomentum(...)
model = MultiAssetWSPR(...)
# ... training code ...

# Save with ModelManager
manager = create_model_manager(mode='training')
manager.save_model(
    model=model,
    model_name='wspr_livestock',
    metadata={
        'tickers': multi_asset.tickers,
        'config': {...},
    },
)

# Load for inference
manager_prod = create_model_manager(mode='production', use_s3=True, s3_config={...})
model, metadata = manager_prod.load_model('wspr_livestock', model_class=MultiAssetWSPR)
```

### Troubleshooting

**S3 Connection Issues:**
- Verify AWS credentials
- Check bucket permissions
- Ensure bucket exists in specified region
- Test with `aws s3 ls s3://your-bucket/` command

**Model Loading Errors:**
- Ensure model class is imported
- Check PyTorch version compatibility
- Verify state dict keys match model architecture

**Cache Issues:**
- Cache location: `~/.ctaflow/cache/s3/`
- Clear cache: `rm -rf ~/.ctaflow/cache/s3/`
- Disable cache: `download_file(..., use_cache=False)`

### See Also

- Main CLI: `main.py` in CTAFlow root
- Dashboard integration: `app/utils/` for Dash frontend
- Training examples: `notebooks/multi_ticker_training.ipynb`
