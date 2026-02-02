"""
Example: Saving Model Predictions to Parquet

This script demonstrates how to save predictions from a trained
MultiAssetWSPR model in the format expected by the dashboard.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add CTAFlow to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from CTAFlow.models.multi_asset import MultiAssetMomentum
from CTAFlow.models.deep_learning.multi_branch.dual_model import MultiAssetWSPR
from app.utils import DataLoader


def load_trained_model(checkpoint_path: str, device: str = 'cuda'):
    """Load a trained MultiAssetWSPR model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = MultiAssetWSPR(
        f_sum=config['f_sum'],
        f_profile=config['f_profile'],
        f_raster=config['f_raster'],
        f_seq=config['f_seq'],
        d_model=config['d_model'],
        sum_lstm_hidden=config['sum_lstm_hidden'],
        prof_lstm_hidden=config['prof_lstm_hidden'],
        meta_hidden=config['meta_hidden'],
        n_tickers=config['n_tickers'],
        n_asset_classes=config['n_asset_classes'],
        n_asset_subclasses=config['n_asset_subclasses'],
        task=config['task'],
        num_classes=config.get('num_classes', 3),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint


def generate_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
    task: str = 'classification',
) -> pd.DataFrame:
    """
    Generate predictions from model and format for dashboard.

    Returns DataFrame with columns:
    - datetime: Timestamp
    - ticker: Ticker symbol
    - prediction: Model prediction (class or value)
    - probability_0, probability_1, probability_2: Class probabilities (classification)
    - target: Actual target (if available)
    """
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            # Unpack batch (format depends on collate_fn)
            summary_days, profile_days, raster_recent, seq_recent, seq_lens, targets, meta = batch

            # Move to device
            summary_days = summary_days.to(device)
            profile_days = profile_days.to(device)
            raster_recent = raster_recent.to(device)
            seq_recent = seq_recent.to(device)
            seq_lens = seq_lens.to(device)
            targets = targets.to(device)
            meta = {k: v.to(device) for k, v in meta.items()}

            # Forward pass
            outputs = model(
                summary_days=summary_days,
                profile_days=profile_days,
                raster_recent=raster_recent,
                seq_recent=seq_recent,
                seq_lens_recent=seq_lens,
                meta=meta,
            )

            # Process outputs
            if task == 'classification':
                probs = torch.softmax(outputs, dim=1)
                pred_classes = probs.argmax(dim=1)

                # Convert to numpy
                probs_np = probs.cpu().numpy()
                pred_classes_np = pred_classes.cpu().numpy()
            else:
                # Regression
                pred_values_np = outputs.squeeze().cpu().numpy()

            targets_np = targets.cpu().numpy()
            ticker_ids_np = meta['ticker_id'].cpu().numpy()

            # Build predictions list
            batch_size = len(targets_np)
            for i in range(batch_size):
                pred_dict = {
                    'datetime': datetime.now(),  # Replace with actual timestamp from data
                    'ticker_id': int(ticker_ids_np[i]),
                    'target': float(targets_np[i]),
                }

                if task == 'classification':
                    pred_dict['prediction'] = int(pred_classes_np[i])
                    for c in range(probs_np.shape[1]):
                        pred_dict[f'probability_{c}'] = float(probs_np[i, c])
                else:
                    pred_dict['prediction'] = float(pred_values_np[i])

                predictions.append(pred_dict)

    return pd.DataFrame(predictions)


def main():
    """
    Example workflow: Load model, generate predictions, save to Parquet.
    """
    # Configuration
    MODEL_PATH = 'results/multi_ticker_wspr/multi_asset_wspr_HE_LE.pth'
    DATA_ROOT = Path('F:/Upload/s3')  # Or your data location
    TICKERS = ['HE', 'LE']
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Ticker ID mapping (should match training)
    TICKER_ID_MAP = {'HE': 0, 'LE': 1}
    TICKER_NAME_MAP = {0: 'HE', 1: 'LE'}

    print(f"Loading model from {MODEL_PATH}")
    model, checkpoint = load_trained_model(MODEL_PATH, device=DEVICE)

    print(f"Model task: {checkpoint['config']['task']}")
    print(f"Tickers: {checkpoint['tickers']}")

    # Load data using MultiAssetMomentum
    from CTAFlow.models.multi_asset import GenericFiles, SummarySelectionConfig

    multi_asset = MultiAssetMomentum(
        root_dir=DATA_ROOT,
        tickers=TICKERS,
        generic_files=GenericFiles(),
        summary_config=SummarySelectionConfig(strategy='exact'),
    )

    # Get validation loader
    _, val_loader = multi_asset.get_loaders(
        tickers=TICKERS,
        mode='concat',
        use_wspr=True,
        ticker_id_map=TICKER_ID_MAP,
        asset_class_id_map={'HE': 0, 'LE': 0},
        asset_subclass_id_map={'HE': 0, 'LE': 1},
        val_split=True,
        val_split_size=0.2,
        windowed=True,
        window_days=checkpoint['config']['window_size'],
        include_spatial=True,
        use_rasterized=True,
        batch_size=32,
    )

    print(f"\nGenerating predictions on {len(val_loader.dataset)} validation samples")
    predictions_df = generate_predictions(
        model=model,
        data_loader=val_loader,
        device=DEVICE,
        task=checkpoint['config']['task'],
    )

    # Map ticker IDs to names
    predictions_df['ticker'] = predictions_df['ticker_id'].map(TICKER_NAME_MAP)

    print(f"\nGenerated {len(predictions_df)} predictions")
    print("\nSample predictions:")
    print(predictions_df.head())

    # Save predictions using DataLoader
    data_loader = DataLoader(
        use_s3=False,  # Set to True to upload to S3
        results_dir='app/results',
    )

    # Save per-ticker predictions
    for ticker in TICKERS:
        ticker_preds = predictions_df[predictions_df['ticker'] == ticker].copy()

        if len(ticker_preds) > 0:
            output_path = data_loader.save_predictions(
                df=ticker_preds,
                model_name='wspr_HE_LE',
                ticker=ticker,
                upload_to_s3=False,  # Set to True to upload
            )
            print(f"\nSaved {len(ticker_preds)} predictions for {ticker} to {output_path}")

    # Save combined predictions
    all_output_path = data_loader.save_predictions(
        df=predictions_df,
        model_name='wspr_HE_LE',
        ticker=None,  # All tickers
        upload_to_s3=False,
    )
    print(f"\nSaved {len(predictions_df)} combined predictions to {all_output_path}")

    print("\nDone! Predictions saved to app/results/predictions/")
    print("You can now view them in the dashboard.")


if __name__ == '__main__':
    main()
