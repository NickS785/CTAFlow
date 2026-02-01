"""Parquet storage utilities for model results and intermediate data.

This module provides a unified interface for reading and writing Parquet files,
which is the preferred storage format for CTAFlow model outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

# Default results directory
DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent.parent / "results"


class ParquetStore:
    """Parquet storage manager for model results.

    Parameters
    ----------
    base_path : Path or str, optional
        Base directory for Parquet files. Defaults to app/results.
    """

    def __init__(self, base_path: Optional[Union[Path, str]] = None) -> None:
        self.base_path = Path(base_path) if base_path else DEFAULT_RESULTS_PATH
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        partition_cols: Optional[List[str]] = None,
        compression: str = "snappy",
    ) -> Path:
        """Save a DataFrame to Parquet format.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to save.
        name : str
            Name for the Parquet file (without extension).
        partition_cols : List[str], optional
            Columns to partition by (creates directory structure).
        compression : str
            Compression algorithm. Default is 'snappy'.

        Returns
        -------
        Path
            Path to the saved Parquet file or directory.
        """
        file_path = self.base_path / f"{name}.parquet"

        if partition_cols:
            # Partitioned write creates a directory
            df.to_parquet(
                file_path,
                partition_cols=partition_cols,
                compression=compression,
                engine="pyarrow",
            )
        else:
            df.to_parquet(
                file_path,
                compression=compression,
                engine="pyarrow",
            )

        return file_path

    def load(
        self,
        name: str,
        *,
        columns: Optional[List[str]] = None,
        filters: Optional[List] = None,
    ) -> pd.DataFrame:
        """Load a DataFrame from Parquet format.

        Parameters
        ----------
        name : str
            Name of the Parquet file (without extension).
        columns : List[str], optional
            Specific columns to load. Loads all if None.
        filters : List, optional
            Row group filters for partitioned data.

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame.
        """
        file_path = self.base_path / f"{name}.parquet"

        return pd.read_parquet(
            file_path,
            columns=columns,
            filters=filters,
            engine="pyarrow",
        )

    def list_results(self, pattern: str = "*.parquet") -> List[Path]:
        """List available Parquet files.

        Parameters
        ----------
        pattern : str
            Glob pattern for matching files.

        Returns
        -------
        List[Path]
            List of matching Parquet file paths.
        """
        return list(self.base_path.glob(pattern))

    def delete(self, name: str) -> bool:
        """Delete a Parquet file.

        Parameters
        ----------
        name : str
            Name of the Parquet file (without extension).

        Returns
        -------
        bool
            True if file was deleted, False if it didn't exist.
        """
        file_path = self.base_path / f"{name}.parquet"

        if file_path.exists():
            if file_path.is_dir():
                import shutil

                shutil.rmtree(file_path)
            else:
                file_path.unlink()
            return True
        return False

    def save_model_results(
        self,
        results: Dict[str, pd.DataFrame],
        model_name: str,
        *,
        compression: str = "snappy",
    ) -> Dict[str, Path]:
        """Save multiple DataFrames as model results.

        Parameters
        ----------
        results : Dict[str, pd.DataFrame]
            Dictionary mapping result names to DataFrames.
        model_name : str
            Name of the model (used as prefix).
        compression : str
            Compression algorithm.

        Returns
        -------
        Dict[str, Path]
            Dictionary mapping result names to saved file paths.
        """
        saved_paths = {}
        for result_name, df in results.items():
            full_name = f"{model_name}_{result_name}"
            saved_paths[result_name] = self.save(
                df, full_name, compression=compression
            )
        return saved_paths
