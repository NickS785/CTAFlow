"""Initialize ArcticDB libraries for CTAFlow data storage.

This module provides utilities to create and verify Arctic libraries for:
- Intraday data (from .scid files)
- Daily market data (from DLY files)
- Forward curve data (from DLY files)
- COT data

Run this module to initialize all libraries before first use.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

try:
    from arcticdb import Arctic
    HAS_ARCTICDB = True
except ImportError:
    HAS_ARCTICDB = False
    Arctic = None

from ..config import (
    INTRADAY_ADB_PATH,
    DAILY_ADB_PATH,
    CURVE_ADB_PATH,
    COT_ADB_PATH,
)

logger = logging.getLogger(__name__)


def create_arctic_library(uri: str, library_name: str = "default") -> bool:
    """
    Create an Arctic library if it doesn't exist.

    Parameters:
    -----------
    uri : str
        Arctic URI (e.g., 'lmdb://F:/Data/intraday')
    library_name : str
        Library name to create (default: 'default')

    Returns:
    --------
    bool
        True if library was created or already exists, False on error
    """
    if not HAS_ARCTICDB:
        logger.error("ArcticDB not installed. Install with: pip install arcticdb")
        return False

    try:
        arctic = Arctic(uri)

        # Check if library exists
        if library_name in arctic.list_libraries():
            logger.info(f"Library '{library_name}' already exists at {uri}")
            return True

        # Create library
        arctic.create_library(library_name)
        logger.info(f"Created library '{library_name}' at {uri}")
        return True

    except Exception as e:
        logger.error(f"Failed to create library '{library_name}' at {uri}: {e}")
        return False


def initialize_all_libraries() -> Dict[str, bool]:
    """
    Initialize all ArcticDB libraries for CTAFlow.

    Creates the following libraries:
    - Intraday: lmdb://F:/Data/intraday
    - Daily: lmdb://F:/Data/daily
    - Curves: lmdb://F:/Data/curves
    - COT: lmdb://F:/Data/cot

    Returns:
    --------
    Dict[str, bool]
        Map of library URI to creation success status
    """
    if not HAS_ARCTICDB:
        logger.error("ArcticDB not installed. Cannot initialize libraries.")
        return {}

    libraries = {
        "intraday": INTRADAY_ADB_PATH,
        "daily": DAILY_ADB_PATH,
        "curves": CURVE_ADB_PATH,
        "cot": COT_ADB_PATH,
    }

    results = {}
    for name, uri in libraries.items():
        logger.info(f"Initializing {name} library at {uri}...")
        success = create_arctic_library(uri, library_name="default")
        results[uri] = success

    # Summary
    successful = sum(results.values())
    total = len(results)
    logger.info(f"Library initialization: {successful}/{total} successful")

    return results


def verify_libraries() -> Dict[str, Dict[str, any]]:
    """
    Verify all Arctic libraries are accessible and report statistics.

    Returns:
    --------
    Dict[str, Dict[str, any]]
        Map of library name to verification results with keys:
        - accessible: bool
        - library_count: int
        - symbols: List[str]
        - error: Optional[str]
    """
    if not HAS_ARCTICDB:
        return {"error": "ArcticDB not installed"}

    libraries = {
        "intraday": INTRADAY_ADB_PATH,
        "daily": DAILY_ADB_PATH,
        "curves": CURVE_ADB_PATH,
        "cot": COT_ADB_PATH,
    }

    results = {}
    for name, uri in libraries.items():
        try:
            arctic = Arctic(uri)
            lib_list = arctic.list_libraries()

            # Try to access default library
            symbols = []
            if "default" in lib_list:
                lib = arctic["default"]
                symbols = lib.list_symbols()

            results[name] = {
                "accessible": True,
                "uri": uri,
                "library_count": len(lib_list),
                "libraries": lib_list,
                "symbols": symbols,
                "symbol_count": len(symbols),
            }

        except Exception as e:
            results[name] = {
                "accessible": False,
                "uri": uri,
                "error": str(e),
            }

    return results


def print_library_status():
    """Print formatted status of all Arctic libraries."""
    print("\n" + "="*70)
    print("CTAFlow ArcticDB Library Status")
    print("="*70 + "\n")

    status = verify_libraries()

    if "error" in status:
        print(f"ERROR: {status['error']}")
        return

    for name, info in status.items():
        print(f"{name.upper()} Library")
        print("-" * 70)
        print(f"  URI: {info['uri']}")

        if info["accessible"]:
            print(f"  Status: ✓ Accessible")
            print(f"  Libraries: {info['library_count']}")
            print(f"  Symbols: {info['symbol_count']}")

            if info['symbols']:
                print(f"  Sample symbols: {', '.join(info['symbols'][:5])}")
                if len(info['symbols']) > 5:
                    print(f"    ... and {len(info['symbols']) - 5} more")
        else:
            print(f"  Status: ✗ Error")
            print(f"  Error: {info.get('error', 'Unknown')}")

        print()


def main():
    """Initialize and verify Arctic libraries."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\nInitializing CTAFlow ArcticDB libraries...\n")

    # Initialize
    results = initialize_all_libraries()

    # Verify
    print_library_status()

    # Return exit code
    if all(results.values()):
        print("✓ All libraries initialized successfully!")
        return 0
    else:
        failed = [uri for uri, success in results.items() if not success]
        print(f"✗ Failed to initialize: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
