# CTAFlow Dashboard Setup Summary

## Completed Tasks

### 1. Updated CLAUDE.md Documentation
Added comprehensive documentation for WSPR models in `CTAFlow/models/deep_learning/multi_branch/dual_model.py`:
- **RecurrentWSPR**: 5-path windowed model for single-ticker orderflow prediction
- **MultiAssetWSPR**: 6-path multi-ticker model with MetaModalityEncoder for cross-asset training

### 2. Created App Directory Structure

```
app/
├── __init__.py                     # Package initialization
├── app.py                          # Main Dash application
├── requirements.txt                # Dashboard dependencies
├── README.md                       # Comprehensive documentation
├── .env.example                    # Environment variable template
├── SETUP_SUMMARY.md               # This file
│
├── components/                     # Dash components (to be implemented)
│   ├── header.py
│   ├── model_selector.py
│   └── performance_charts.py
│
├── assets/                         # Static assets
│   └── custom.css                 # Custom styling
│
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── s3_client.py               # AWS S3 integration
│   ├── parquet_handler.py         # Parquet file operations
│   └── data_loader.py             # Unified data loading interface
│
├── examples/                       # Example scripts
│   └── save_predictions_example.py # How to save predictions
│
└── results/                        # Local results storage (Parquet)
    ├── models/                     # Cached model checkpoints
    ├── predictions/                # Prediction results
    ├── backtests/                  # Backtest outputs
    └── metrics/                    # Training metrics
```

### 3. AWS S3 Integration (`app/utils/s3_client.py`)

**Features:**
- Download/upload files to S3
- List available models and data
- Download complete ticker datasets
- Automatic credential handling from environment variables

**Key Methods:**
```python
s3 = S3Client(bucket_name='ctaflow-data')
s3.list_models()  # List all model checkpoints
s3.download_ticker_data('HE')  # Download all HE data files
s3.download_file('models/wspr_HE_LE.pth', 'local/path/')
```

### 4. Parquet Storage Handler (`app/utils/parquet_handler.py`)

**Features:**
- Efficient columnar storage with compression
- Save/load predictions, backtests, and metrics
- Date-based filtering for fast queries
- Automatic schema management

**Key Methods:**
```python
handler = ParquetHandler('app/results')
handler.save_predictions(df, model_name='wspr_HE_LE', ticker='HE')
df = handler.load_predictions('wspr_HE_LE', date_range=('2024-01-01', '2024-12-31'))
handler.save_metrics(metrics_dict, model_name='wspr_HE_LE', split='validation')
```

### 5. Unified Data Loader (`app/utils/data_loader.py`)

**Features:**
- Combines S3 and local Parquet handling
- Automatic caching of S3 downloads
- List available models from S3 and cache
- Load/save predictions with optional S3 upload

**Key Methods:**
```python
loader = DataLoader(use_s3=True, cache_dir='./cache')
loader.load_ticker_data('HE')  # Auto-downloads from S3 if not cached
loader.load_model_checkpoint('wspr_HE_LE')  # Downloads model
loader.save_predictions(df, 'wspr_HE_LE', ticker='HE', upload_to_s3=True)
```

### 6. Main Dashboard Application (`app/app.py`)

**Current Features:**
- Bootstrap-based responsive layout
- Model and ticker selection dropdowns
- Placeholder visualizations for:
  - Performance overview
  - Predictions chart
  - Trading metrics display
  - Recent activity feed
- Auto-refresh interval component (disabled by default)

**To Be Implemented:**
- Callbacks for interactive updates
- Component integration
- Real-time data loading
- Performance charts

## Data Flow

### Training → Dashboard

1. **Train Model**: Use `MultiAssetMomentum` to train multi-ticker WSPR model
2. **Save Checkpoint**: Model saved with metadata (tickers, config, history)
3. **Generate Predictions**: Run inference on validation/test data
4. **Save to Parquet**: Use `DataLoader.save_predictions()` to save in dashboard format
5. **Upload to S3** (optional): Set `upload_to_s3=True` to sync with cloud storage

### Dashboard → Display

1. **Load Models**: Dashboard lists available models from S3/cache
2. **Select Model**: User selects model from dropdown
3. **Load Data**: Dashboard loads predictions/metrics from Parquet
4. **Visualize**: Display performance charts, metrics, and predictions
5. **Analyze**: Interactive filtering by date, ticker, performance metrics

## Expected S3 Bucket Structure

```
s3://ctaflow-data/
├── models/                         # Trained model checkpoints
│   ├── wspr_HE_LE.pth
│   ├── wspr_CL_RB.pth
│   └── tri_modal_HE.pth
│
├── data/                           # Raw ticker data
│   ├── HE/
│   │   ├── features.csv            # Summary features
│   │   ├── profiles.npz            # Market profile arrays
│   │   ├── vpin.parquet            # Sequential VPIN data
│   │   ├── rasterized.npz          # Rasterized VPIN
│   │   ├── target.csv              # Target values
│   │   └── intraday.csv            # Intraday bars
│   ├── LE/
│   │   └── ...
│   └── {TICKER}/
│       └── ...
│
└── predictions/                    # Model predictions (optional cloud storage)
    ├── wspr_HE_LE_HE.parquet
    ├── wspr_HE_LE_LE.parquet
    └── ...
```

## Prediction Parquet Schema

### Classification Model
| Column | Type | Description |
|--------|------|-------------|
| datetime | datetime64 | Timestamp of prediction |
| ticker | string | Ticker symbol |
| prediction | int64 | Predicted class (0, 1, 2) |
| probability_0 | float64 | Class 0 probability |
| probability_1 | float64 | Class 1 probability |
| probability_2 | float64 | Class 2 probability |
| target | int64 | Actual class (if known) |
| pnl | float64 | Realized PnL (if available) |

### Regression Model
| Column | Type | Description |
|--------|------|-------------|
| datetime | datetime64 | Timestamp of prediction |
| ticker | string | Ticker symbol |
| prediction | float64 | Predicted return |
| target | float64 | Actual return (if known) |
| pnl | float64 | Realized PnL (if available) |

## Next Steps

### High Priority
1. **Implement Dashboard Components**:
   - `components/header.py`: Navigation and model info
   - `components/model_selector.py`: Model/ticker selection with metadata
   - `components/performance_charts.py`: PnL, Sharpe, drawdown charts

2. **Add Callbacks**:
   - Load models list on startup
   - Update charts when model/ticker selected
   - Filter by date range
   - Calculate and display trading metrics

3. **Testing**:
   - Test S3 integration with real credentials
   - Generate sample predictions from trained models
   - Verify Parquet read/write performance
   - Test dashboard with real data

### Medium Priority
4. **Additional Features**:
   - Backtest visualization
   - Live prediction mode
   - Model comparison view
   - Export results to PDF/Excel

5. **Deployment**:
   - Dockerize the application
   - Set up CI/CD pipeline
   - Deploy to cloud (AWS ECS, Google Cloud Run, etc.)

### Low Priority
6. **Enhancements**:
   - User authentication
   - Multiple user workspaces
   - Advanced filtering and search
   - Real-time updates via WebSocket

## Configuration

### Environment Variables (.env)

```bash
# Required for S3
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
CTAFLOW_S3_BUCKET=ctaflow-data

# Optional
DASH_DEBUG=True
DASH_HOST=0.0.0.0
DASH_PORT=8050
RESULTS_DIR=app/results
CACHE_DIR=data_cache
```

### Install Dependencies

```bash
cd app/
pip install -r requirements.txt
```

### Run Dashboard

```bash
# Development
python app.py

# Production
gunicorn app:server -b 0.0.0.0:8050 --workers 4
```

## Example Usage

See `app/examples/save_predictions_example.py` for a complete workflow:
1. Load trained MultiAssetWSPR model
2. Generate predictions on validation data
3. Format predictions DataFrame
4. Save to Parquet (locally and/or S3)

## Notes

- **Parquet Benefits**: 10-100x faster than CSV for large datasets, built-in compression, schema enforcement
- **S3 Caching**: Downloads are cached locally to avoid repeated S3 API calls
- **Modular Design**: Components can be developed and tested independently
- **Scalability**: Designed to handle multi-ticker, multi-model deployments

## References

- Main training notebook: `notebooks/multi_ticker_training.ipynb`
- WSPR model architecture: `CTAFlow/models/deep_learning/multi_branch/dual_model.py`
- Multi-asset training: `CTAFlow/models/multi_asset.py`
- Training loops: `CTAFlow/models/deep_learning/training/loops.py`
