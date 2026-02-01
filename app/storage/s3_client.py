"""AWS S3 client for downloading model data.

This module provides utilities for downloading data stored on AWS S3 that is
used by CTAFlow models. Credentials are loaded from environment variables
or the standard AWS credentials file (~/.aws/credentials).

Environment Variables
--------------------
AWS_ACCESS_KEY_ID : str
    AWS access key ID.
AWS_SECRET_ACCESS_KEY : str
    AWS secret access key.
AWS_DEFAULT_REGION : str, optional
    Default AWS region. Defaults to 'us-east-1'.
CTAFLOW_S3_BUCKET : str, optional
    Default S3 bucket name for CTAFlow data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception


DEFAULT_BUCKET = os.environ.get("CTAFLOW_S3_BUCKET", "ctaflow-data")
DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


class S3DataClient:
    """S3 client for downloading CTAFlow model data.

    Parameters
    ----------
    bucket : str, optional
        S3 bucket name. Defaults to CTAFLOW_S3_BUCKET env var or 'ctaflow-data'.
    region : str, optional
        AWS region. Defaults to AWS_DEFAULT_REGION env var or 'us-east-1'.
    local_cache_path : Path or str, optional
        Local directory for caching downloaded files.
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        region: Optional[str] = None,
        local_cache_path: Optional[Union[Path, str]] = None,
    ) -> None:
        if not HAS_BOTO3:
            raise ImportError(
                "boto3 is required for S3 support. Install with: pip install boto3"
            )

        self.bucket = bucket or DEFAULT_BUCKET
        self.region = region or DEFAULT_REGION
        self.local_cache_path = (
            Path(local_cache_path)
            if local_cache_path
            else Path(__file__).resolve().parent.parent / "results" / ".s3_cache"
        )
        self.local_cache_path.mkdir(parents=True, exist_ok=True)

        self._client = None
        self._resource = None

    @property
    def client(self):
        """Lazily initialize and return the S3 client."""
        if self._client is None:
            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    @property
    def resource(self):
        """Lazily initialize and return the S3 resource."""
        if self._resource is None:
            self._resource = boto3.resource("s3", region_name=self.region)
        return self._resource

    def download_file(
        self,
        s3_key: str,
        local_path: Optional[Union[Path, str]] = None,
        *,
        use_cache: bool = True,
    ) -> Path:
        """Download a file from S3.

        Parameters
        ----------
        s3_key : str
            S3 object key (path within the bucket).
        local_path : Path or str, optional
            Local path to save the file. Defaults to cache directory.
        use_cache : bool
            If True, skip download if file exists locally.

        Returns
        -------
        Path
            Path to the downloaded file.

        Raises
        ------
        FileNotFoundError
            If the S3 object does not exist.
        PermissionError
            If AWS credentials are invalid or missing.
        """
        if local_path is None:
            local_path = self.local_cache_path / s3_key.replace("/", "_")
        else:
            local_path = Path(local_path)

        if use_cache and local_path.exists():
            return local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.client.download_file(self.bucket, s3_key, str(local_path))
        except NoCredentialsError as e:
            raise PermissionError(
                "AWS credentials not found. Set AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY environment variables or configure "
                "~/.aws/credentials"
            ) from e
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                raise FileNotFoundError(
                    f"S3 object not found: s3://{self.bucket}/{s3_key}"
                ) from e
            raise

        return local_path

    def download_parquet(
        self,
        s3_key: str,
        *,
        use_cache: bool = True,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Download and read a Parquet file from S3.

        Parameters
        ----------
        s3_key : str
            S3 object key for the Parquet file.
        use_cache : bool
            If True, use cached file if available.
        columns : List[str], optional
            Specific columns to load.

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame.
        """
        local_path = self.download_file(s3_key, use_cache=use_cache)
        return pd.read_parquet(local_path, columns=columns, engine="pyarrow")

    def list_objects(
        self, prefix: str = "", suffix: str = ""
    ) -> List[str]:
        """List objects in the S3 bucket.

        Parameters
        ----------
        prefix : str
            Filter objects by key prefix.
        suffix : str
            Filter objects by key suffix.

        Returns
        -------
        List[str]
            List of S3 object keys.
        """
        paginator = self.client.get_paginator("list_objects_v2")
        keys = []

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if suffix and not key.endswith(suffix):
                    continue
                keys.append(key)

        return keys

    def list_parquet_files(self, prefix: str = "") -> List[str]:
        """List Parquet files in the S3 bucket.

        Parameters
        ----------
        prefix : str
            Filter by key prefix.

        Returns
        -------
        List[str]
            List of Parquet file keys.
        """
        return self.list_objects(prefix=prefix, suffix=".parquet")

    def upload_file(
        self,
        local_path: Union[Path, str],
        s3_key: str,
    ) -> str:
        """Upload a file to S3.

        Parameters
        ----------
        local_path : Path or str
            Local file path to upload.
        s3_key : str
            S3 object key (destination path).

        Returns
        -------
        str
            S3 URI of the uploaded file.
        """
        self.client.upload_file(str(local_path), self.bucket, s3_key)
        return f"s3://{self.bucket}/{s3_key}"

    def upload_parquet(
        self,
        df: pd.DataFrame,
        s3_key: str,
        *,
        compression: str = "snappy",
    ) -> str:
        """Save a DataFrame to Parquet and upload to S3.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to upload.
        s3_key : str
            S3 object key for the Parquet file.
        compression : str
            Compression algorithm.

        Returns
        -------
        str
            S3 URI of the uploaded file.
        """
        local_path = self.local_cache_path / s3_key.replace("/", "_")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(local_path, compression=compression, engine="pyarrow")
        return self.upload_file(local_path, s3_key)

    def clear_cache(self) -> int:
        """Clear the local cache directory.

        Returns
        -------
        int
            Number of files deleted.
        """
        import shutil

        count = sum(1 for _ in self.local_cache_path.iterdir())
        shutil.rmtree(self.local_cache_path)
        self.local_cache_path.mkdir(parents=True, exist_ok=True)
        return count
