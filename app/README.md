# CTAFlow Dashboard

A Dash-based web dashboard for visualizing and analyzing CTAFlow trading models, predictions, and performance metrics.

## Features

- **Multi-Asset Model Support**: View performance across multiple tickers
- **Real-time Predictions**: Display and analyze model predictions
- **Backtest Analysis**: Comprehensive backtesting results and metrics
- **AWS S3 Integration**: Automatic data and model syncing
- **Parquet Storage**: Efficient columnar storage for predictions and metrics

## Directory Structure

```
app/
├── app.py                  # Main Dash application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .env.example           # Environment variable template
├── components/            # Dash components
│   ├── header.py
│   ├── model_selector.py
│   └── performance_charts.py
├── utils/                 # Utility modules
│   ├── s3_client.py      # AWS S3 integration
│   ├── parquet_handler.py # Parquet file operations
│   └── data_loader.py    # Unified data loading
├── assets/               # Static assets (CSS, images)
│   └── custom.css
└── results/              # Local results storage
    ├── models/           # Cached model checkpoints
    ├── predictions/      # Prediction results (Parquet)
    ├── backtests/        # Backtest outputs (Parquet)
    └── metrics/          # Training metrics (Parquet)
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
# AWS S3 Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
CTAFLOW_S3_BUCKET=ctaflow-data

# Optional: Dash configuration
DASH_DEBUG=True
DASH_HOST=0.0.0.0
DASH_PORT=8050
```

### 3. Run the Dashboard

**Development:**
```bash
python app.py
```

**Production (with Gunicorn):**
```bash
gunicorn app:server -b 0.0.0.0:8050 --workers 4
```

Dashboard will be available at `http://localhost:8050`

## Usage

### Loading Data from S3

The dashboard automatically syncs with AWS S3 when enabled:

```python
from CTAFlow.data.utils import DataLoader

# Initialize with S3
loader = DataLoader(use_s3=True)

# Download ticker data
data_files = loader.load_ticker_data('HE')  # Downloads to cache
model_path = loader.load_model_checkpoint('wspr_HE_LE')  # Downloads model
```

### Saving Predictions

```python
import pandas as pd
from CTAFlow.data.utils import DataLoader

loader = DataLoader(use_s3=True, results_dir='app/results')

# Create predictions DataFrame
predictions = pd.DataFrame({
    'datetime': dates,
    'ticker': tickers,
    'prediction': preds,
    'probability_0': probs[:, 0],
    'probability_1': probs[:, 1],
    'probability_2': probs[:, 2],
})

# Save locally (Parquet)
loader.save_predictions(predictions, model_name='wspr_HE_LE', ticker='HE')

# Save and upload to S3
loader.save_predictions(predictions, model_name='wspr_HE_LE',
                        ticker='HE', upload_to_s3=True)
```

### Reading Predictions

```python
# Load all predictions for a model
df = loader.load_predictions('wspr_HE_LE', ticker='HE')

# Load with date filtering
df = loader.load_predictions(
    'wspr_HE_LE',
    ticker='HE',
    date_range=('2024-01-01', '2024-12-31')
)
```

## S3 Bucket Structure

Expected S3 bucket organization:

```
s3://ctaflow-data/
├── models/
│   ├── wspr_HE_LE.pth
│   └── wspr_CL_RB.pth
├── data/
│   ├── HE/
│   │   ├── features.csv
│   │   ├── profiles.npz
│   │   ├── vpin.parquet
│   │   ├── rasterized.npz
│   │   ├── target.csv
│   │   └── intraday.csv
│   └── LE/
│       └── ...
└── predictions/
    ├── wspr_HE_LE_HE.parquet
    └── wspr_HE_LE_LE.parquet
```

## Data Format: Predictions Parquet

Expected schema for prediction Parquet files:

| Column | Type | Description |
|--------|------|-------------|
| datetime | datetime64 | Timestamp |
| ticker | string | Ticker symbol |
| prediction | int/float | Model prediction (class or value) |
| probability_0 | float | Class 0 probability (classification) |
| probability_1 | float | Class 1 probability (classification) |
| probability_2 | float | Class 2 probability (classification) |
| target | float | Actual target (if known) |
| pnl | float | Realized PnL (if available) |

## Performance Monitoring

The dashboard tracks:
- **Classification metrics**: Accuracy, directional accuracy, confusion matrix
- **Trading metrics**: PnL, Sharpe ratio, max drawdown, win rate
- **Regression metrics**: MSE, MAE, R², directional accuracy

## Development

### Adding New Components

Create new components in `app/components/`:

```python
# app/components/my_component.py
from dash import html, dcc

def create_my_component():
    return html.Div([
        html.H3("My Component"),
        dcc.Graph(id='my-chart'),
    ])
```

Import in `app.py` and add to layout.

### Adding New Callbacks

Define callbacks in `app.py`:

```python
from dash import Input, Output

@app.callback(
    Output('my-chart', 'figure'),
    Input('model-selector', 'value')
)
def update_chart(model_name):
    # Load and process data
    return figure
```

## Deployment

### Docker Deployment (Recommended)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY app/ /app/
RUN pip install -r requirements.txt

CMD ["gunicorn", "app:server", "-b", "0.0.0.0:8050", "--workers", "4"]
```

Build and run:
```bash
docker build -t ctaflow-dashboard .
docker run -p 8050:8050 --env-file .env ctaflow-dashboard
```

### Cloud Deployment

The dashboard can be deployed to:
- AWS EC2 (with ECS/EKS for containers)
- Google Cloud Run
- Heroku
- Azure App Service

Ensure environment variables are configured in your cloud platform.

## Troubleshooting

**S3 Connection Issues:**
- Verify AWS credentials in `.env`
- Check S3 bucket permissions
- Ensure bucket exists and region is correct

**Parquet Read Errors:**
- Verify Parquet file format with `pyarrow`
- Check file permissions
- Ensure compatible pyarrow version

**Dash Not Loading:**
- Check port 8050 is not in use
- Verify all dependencies installed
- Check console for Python errors

## License

See main CTAFlow LICENSE file.
