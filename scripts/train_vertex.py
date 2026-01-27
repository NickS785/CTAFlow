
import argparse
import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from google.cloud import storage

from CTAFlow.models.intraday_momentum import DeepIDMomentum
from CTAFlow.models.deep_learning.multi_branch.dual_model import RecurrentWSPR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_gcs_directory(bucket_name, source_directory, destination_directory):
    """Downloads a directory from GCS to a local path."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_directory)  # Get all files in the directory

    for blob in blobs:
        # Create local directory structure
        dest_file_path = os.path.join(destination_directory, os.path.relpath(blob.name, source_directory))
        os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
        
        logging.info(f"Downloading gs://{bucket_name}/{blob.name} to {dest_file_path}")
        blob.download_to_filename(dest_file_path)

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    logging.info(f"File {source_file_name} uploaded to gs://{bucket_name}/{destination_blob_name}")


def main(args):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Create local temp directory
    local_data_dir = Path('./tmp_data')
    local_data_dir.mkdir(exist_ok=True, parents=True)

    # Download data from GCS
    bucket_name = args.data_gcs_path.split('/')[2]
    source_directory = '/'.join(args.data_gcs_path.split('/')[3:]) + f"/{args.ticker}"
    download_gcs_directory(bucket_name, source_directory, str(local_data_dir))
    
    # Define local file paths from the downloaded data
    intraday_path = local_data_dir / args.intraday_filename
    features_path = local_data_dir / args.features_filename
    target_path = local_data_dir / args.target_filename
    vpin_path = local_data_dir / args.vpin_filename
    profile_path = local_data_dir / args.profile_filename
    rasterized_path = local_data_dir / args.rasterized_filename

    # Load data using DeepIDMomentum
    logging.info("Loading data...")
    base_model = DeepIDMomentum.from_files(
        intraday_path=str(intraday_path),
        features_path=str(features_path),
        sequential_path=str(vpin_path),
        profile_path=str(profile_path),
        rasterized_path=str(rasterized_path),
        target_path=str(target_path),
    )

    # Normalize features
    logging.info("Normalizing features...")
    base_model.normalize_sequential_features(scale_to_basis_points=True, scale_orderflow=True)
    base_model.scale_summary_data()

    # Get data loaders for RecurrentWSPR
    logging.info("Creating data loaders...")
    train_loader, val_loader = base_model.get_loaders(
        val_split=True,
        val_split_size=0.2,
        batch_size=args.batch_size,
        windowed=True,
        use_rasterized=True, # This enables TriModalWindowDataset
        window_days=args.window_days,
        add_raw_returns=True 
    )

    # Initialize model
    dims = base_model.dims
    model = RecurrentWSPR(
        f_sum=dims.summary_dim,
        f_profile=dims.profile_channels,
        f_raster=dims.raster_channels,
        f_seq=dims.seq_dim,
        d_model=args.d_model,
        sum_lstm_hidden=args.sum_lstm_hidden,
        prof_lstm_hidden=args.prof_lstm_hidden,
        task=args.task,
        num_classes=args.num_classes
    ).to(device)

    logging.info(f"Model created: RecurrentWSPR with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # Loss and optimizer
    if args.task == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # Training loop
    logging.info("Starting training...")
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            summary_days, seq_days, profile_days, raster_days, targets, seq_lens, raw_returns = batch
            
            # Slice data for RecurrentWSPR's multi-path inputs
            summary_days = summary_days.to(device)
            profile_days = profile_days.to(device)
            raster_recent = raster_days[:, -1].to(device)
            seq_recent = seq_days[:, -1].to(device)
            seq_lens_recent = seq_lens[:, -1].to(device)
            
            if args.task == 'regression':
                targets = targets.to(device).float().unsqueeze(1)
            else:
                targets = targets.to(device).long()

            optimizer.zero_grad()
            outputs = model(summary_days, profile_days, raster_recent, seq_recent, seq_lens_recent)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                summary_days, seq_days, profile_days, raster_days, targets, seq_lens, raw_returns = batch

                summary_days = summary_days.to(device)
                profile_days = profile_days.to(device)
                raster_recent = raster_days[:, -1].to(device)
                seq_recent = seq_days[:, -1].to(device)
                seq_lens_recent = seq_lens[:, -1].to(device)

                if args.task == 'regression':
                    targets = targets.to(device).float().unsqueeze(1)
                else:
                    targets = targets.to(device).long()
                
                outputs = model(summary_days, profile_days, raster_recent, seq_recent, seq_lens_recent)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        logging.info(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = "best_model.pt"
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"New best model saved to {model_save_path} with validation loss: {best_val_loss:.6f}")

    # Upload model to GCS
    output_bucket_name = args.output_gcs_path.split('/')[2]
    output_blob_name = '/'.join(args.output_gcs_path.split('/')[3:]) + f'/{args.ticker}/best_model.pt'
    upload_to_gcs(output_bucket_name, "best_model.pt", output_blob_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-gcs-path', type=str, required=True, help='GCS path to the data directory (e.g., gs://bucket/data)')
    parser.add_argument('--output-gcs-path', type=str, required=True, help='GCS path to save the output model (e.g., gs://bucket/models)')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol for training')

    # Data filenames
    parser.add_argument('--intraday-filename', type=str, default='intraday.csv')
    parser.add_argument('--features-filename', type=str, default='features.csv')
    parser.add_argument('--target-filename', type=str, default='target.csv')
    parser.add_argument('--vpin-filename', type=str, default='vpin.parquet')
    parser.add_argument('--profile-filename', type=str, default='profiles.npz')
    parser.add_argument('--rasterized-filename', type=str, default='rasterized.npz')

    # Model Hyperparameters
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--sum_lstm_hidden', type=int, default=64)
    parser.add_argument('--prof_lstm_hidden', type=int, default=128)
    parser.add_argument('--task', type=str, default='regression', choices=['regression', 'classification'])
    parser.add_argument('--num_classes', type=int, default=3)

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--window-days', type=int, default=10)

    args = parser.parse_args()
    main(args)
