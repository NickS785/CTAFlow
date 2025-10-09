"""
ArcticDB Setup for Intraday Market Data

This module initializes and configures ArcticDB with LMDB backend for storing
high-frequency intraday market data. Follows the migration guide at migration_guide.md.

Database Configuration:
- Location: F:\Data\intraday
- Backend: LMDB
- Map Size: 50GB
- Libraries: intraday_market (for OHLCV data), intraday_ticks (for raw tick data)
"""

from pathlib import Path
from typing import List, Dict, Optional
import logging

try:
    import arcticdb as adb
    from arcticdb import Arctic, QueryBuilder
    HAS_ARCTICDB = True
except ImportError:
    HAS_ARCTICDB = False
    Arctic = None
    QueryBuilder = None

from CTAFlow.config import intraday_db_path


# ArcticDB Configuration
ARCTIC_URI = "lmdb://F:/Data/intraday?map_size=100GB"
INTRADAY_PATH = Path("F:/Data/intraday")

# Library definitions
LIBRARIES = {
    "intraday_market": "Resampled intraday market data (1T, 5T, 15T, 1H, etc.)",
    "intraday_ticks": "Raw tick data from SCID files",
    "intraday_volume": "Volume-bucketed data (vol_500, vol_1000, etc.)",
    "intraday_metadata": "Contract metadata, roll dates, and data quality info"
}


logger = logging.getLogger(__name__)


def check_arcticdb_available() -> bool:
    """
    Check if ArcticDB is installed and available.

    Returns:
    --------
    bool
        True if ArcticDB is available, False otherwise
    """
    if not HAS_ARCTICDB:
        logger.error(
            "ArcticDB is not installed. Install with: pip install arcticdb"
        )
        return False
    return True


def ensure_directory_exists(path: Path) -> bool:
    """
    Ensure the directory for ArcticDB exists.

    Args:
        path: Path to the database directory

    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Database directory ready: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False


def initialize_arctic(uri: Optional[str] = None) -> Optional[Arctic]:
    """
    Initialize ArcticDB connection with LMDB backend.

    Args:
        uri: ArcticDB URI (default: ARCTIC_URI from config)

    Returns:
        Arctic instance or None if initialization fails

    Examples:
        >>> ac = initialize_arctic()
        >>> print(ac.list_libraries())
    """
    if not check_arcticdb_available():
        return None

    uri = uri or ARCTIC_URI

    try:
        # Ensure directory exists
        if not ensure_directory_exists(INTRADAY_PATH):
            return None

        # Connect to ArcticDB
        ac = Arctic(uri)
        logger.info(f"Connected to ArcticDB at {uri}")

        return ac

    except Exception as e:
        logger.error(f"Failed to initialize ArcticDB: {e}")
        return None


def create_libraries(ac: Arctic, libraries: Optional[Dict[str, str]] = None) -> Dict[str, bool]:
    """
    Create ArcticDB libraries for intraday data storage.

    Args:
        ac: Arctic instance
        libraries: Dictionary of library names and descriptions (default: LIBRARIES)

    Returns:
        Dictionary mapping library names to creation success status

    Libraries:
        - intraday_market: Resampled OHLCV data at various timeframes
        - intraday_ticks: Raw tick data
        - intraday_volume: Volume-bucketed aggregations
        - intraday_metadata: Metadata about contracts and data quality
    """
    libraries = libraries or LIBRARIES
    results = {}

    existing_libs = set(ac.list_libraries())

    for lib_name, description in libraries.items():
        try:
            if lib_name in existing_libs:
                logger.info(f"Library '{lib_name}' already exists")
                results[lib_name] = True
            else:
                ac.create_library(lib_name)
                logger.info(f"Created library '{lib_name}': {description}")
                results[lib_name] = True
        except Exception as e:
            logger.error(f"Failed to create library '{lib_name}': {e}")
            results[lib_name] = False

    return results


def setup_intraday_db(uri: Optional[str] = None,
                      libraries: Optional[Dict[str, str]] = None,
                      validate: bool = True) -> Optional[Arctic]:
    """
    Complete setup of ArcticDB for intraday market data.

    This is the main entry point for initializing the intraday database.

    Args:
        uri: ArcticDB URI (default: ARCTIC_URI)
        libraries: Custom library definitions (default: LIBRARIES)
        validate: Run validation checks after setup

    Returns:
        Arctic instance if setup successful, None otherwise

    Examples:
        >>> # Basic setup
        >>> ac = setup_intraday_db()

        >>> # Custom setup
        >>> custom_libs = {
        ...     "intraday_1min": "1-minute OHLCV data",
        ...     "intraday_5min": "5-minute OHLCV data"
        ... }
        >>> ac = setup_intraday_db(libraries=custom_libs)
    """
    logger.info("Starting ArcticDB setup for intraday market data")

    # Initialize connection
    ac = initialize_arctic(uri)
    if ac is None:
        logger.error("Failed to initialize ArcticDB connection")
        return None

    # Create libraries
    lib_results = create_libraries(ac, libraries)

    # Check results
    successful = sum(lib_results.values())
    total = len(lib_results)

    if successful < total:
        logger.warning(f"Only {successful}/{total} libraries created successfully")
    else:
        logger.info(f"All {total} libraries created successfully")

    # Validation
    if validate:
        validation_results = validate_setup(ac)
        if not validation_results['overall_success']:
            logger.warning("Setup validation failed")

    logger.info("ArcticDB setup complete")
    return ac


def validate_setup(ac: Arctic) -> Dict[str, any]:
    """
    Validate ArcticDB setup and configuration.

    Args:
        ac: Arctic instance

    Returns:
        Dictionary with validation results

    Checks:
        - All expected libraries exist
        - Libraries are accessible
        - Write/read test
    """
    results = {
        'libraries_exist': {},
        'libraries_accessible': {},
        'write_read_test': None,
        'overall_success': True
    }

    # Check library existence
    existing_libs = set(ac.list_libraries())

    for lib_name in LIBRARIES.keys():
        exists = lib_name in existing_libs
        results['libraries_exist'][lib_name] = exists

        if not exists:
            results['overall_success'] = False
            logger.error(f"Library '{lib_name}' does not exist")
        else:
            # Test accessibility
            try:
                lib = ac[lib_name]
                results['libraries_accessible'][lib_name] = True
            except Exception as e:
                results['libraries_accessible'][lib_name] = False
                results['overall_success'] = False
                logger.error(f"Cannot access library '{lib_name}': {e}")

    # Write/read test on intraday_metadata
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        lib = ac['intraday_metadata']

        # Create test data
        test_symbol = '_test_validation'
        test_df = pd.DataFrame({
            'test_value': np.random.randn(10)
        }, index=pd.date_range(datetime.now(), periods=10, freq='1min'))

        test_metadata = {
            'test': True,
            'created_at': str(datetime.now()),
            'purpose': 'validation'
        }

        # Write test
        lib.write(test_symbol, test_df, metadata=test_metadata)

        # Read test
        item = lib.read(test_symbol)
        read_df = item.data
        read_meta = item.metadata

        # Validate
        write_read_success = (
            len(read_df) == len(test_df) and
            read_meta.get('test') == True
        )

        # Cleanup
        lib.delete(test_symbol)

        results['write_read_test'] = write_read_success
        if not write_read_success:
            results['overall_success'] = False
            logger.error("Write/read validation test failed")
        else:
            logger.info("Write/read validation test passed")

    except Exception as e:
        results['write_read_test'] = False
        results['overall_success'] = False
        logger.error(f"Write/read test failed: {e}")

    return results


def get_library_info(ac: Arctic) -> List[Dict[str, any]]:
    """
    Get information about all libraries in the ArcticDB instance.

    Args:
        ac: Arctic instance

    Returns:
        List of dictionaries with library information
    """
    libraries = []

    for lib_name in ac.list_libraries():
        lib = ac[lib_name]
        symbols = lib.list_symbols()

        info = {
            'name': lib_name,
            'description': LIBRARIES.get(lib_name, 'N/A'),
            'num_symbols': len(symbols),
            'symbols': symbols[:10] if len(symbols) <= 10 else symbols[:10] + ['...']
        }
        libraries.append(info)

    return libraries


def print_setup_info(ac: Arctic) -> None:
    """
    Print setup information to console.

    Args:
        ac: Arctic instance
    """
    print("\n" + "="*70)
    print("ArcticDB Intraday Market Data - Setup Information")
    print("="*70)
    print(f"\nDatabase Location: {INTRADAY_PATH}")
    print(f"Database URI: {ARCTIC_URI}")
    print(f"Map Size: 50GB")
    print(f"\nLibraries:")

    lib_info = get_library_info(ac)
    for info in lib_info:
        print(f"\n  {info['name']}")
        print(f"    Description: {info['description']}")
        print(f"    Symbols: {info['num_symbols']}")
        if info['symbols']:
            print(f"    Examples: {', '.join(str(s) for s in info['symbols'][:5])}")

    print("\n" + "="*70)
    print("Setup complete! Use Arctic(ARCTIC_URI) to connect.")
    print("="*70 + "\n")


def reset_database(ac: Arctic, confirm: bool = False) -> bool:
    """
    Reset the database by deleting all libraries.

    WARNING: This will delete ALL data in the database!

    Args:
        ac: Arctic instance
        confirm: Must be True to proceed (safety check)

    Returns:
        bool: True if reset successful
    """
    if not confirm:
        logger.warning("Reset requires confirm=True parameter")
        return False

    logger.warning("Resetting ArcticDB - deleting all libraries")

    try:
        for lib_name in ac.list_libraries():
            ac.delete_library(lib_name)
            logger.info(f"Deleted library: {lib_name}")

        logger.info("Database reset complete")
        return True

    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        return False


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run setup
    print("\nInitializing ArcticDB for intraday market data...")
    print(f"Location: {INTRADAY_PATH}")
    print(f"Size: 50GB")
    print("\nThis will create the following libraries:")
    for lib_name, desc in LIBRARIES.items():
        print(f"  - {lib_name}: {desc}")

    ac = setup_intraday_db()

    if ac:
        print_setup_info(ac)
        print("\nSetup successful!")
        print("\nUsage example:")
        print("  from CTAFlow.utils.adb_setup import initialize_arctic")
        print("  ac = initialize_arctic()")
        print("  lib = ac['intraday_market']")
        print("  lib.write('CL_F_1T', your_dataframe)")
    else:
        print("\nSetup failed. Check logs for details.")
