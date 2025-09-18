#!/usr/bin/env python3
"""
Refresh Curve Calculations Script (Asynchronous)

This script re-sequentializes existing curve data to apply fixes for:
- Proper DTE calculations using actual contract expiry dates
- Correct contract ordering and sequentialization
- Rolling back/front spread improvements
- 360+ DTE contract maintenance

Features:
- Asynchronous processing for increased speed
- Proper HDF5 locking to prevent write conflicts
- Progress tracking and detailed error reporting
- Validation and verification of written data

Usage:
    python refresh_curve_calculations.py [--symbols CL,NG,ES] [--dly-folder path] [--dry-run]
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import threading

import pandas as pd

# Add the CTAFlow package to the path
sys.path.insert(0, str(Path(__file__).parent))

from CTAFlow.data.data_client import DataClient
from CTAFlow.data.contract_handling.dly_contract_manager import DLYContractManager, DLYFolderUpdater
from CTAFlow.config import DLY_DATA_PATH, MARKET_DATA_PATH

# Global HDF5 lock for thread-safe operations
HDF5_LOCK = asyncio.Lock()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Refresh curve calculations by re-sequentializing existing data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Refresh all available symbols
    python refresh_curve_calculations.py

    # Refresh specific symbols
    python refresh_curve_calculations.py --symbols CL,NG,ES

    # Use custom DLY folder
    python refresh_curve_calculations.py --dly-folder /path/to/dly/files

    # Dry run to see what would be processed
    python refresh_curve_calculations.py --dry-run
        """
    )

    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols to refresh (e.g., CL,NG,ES). If not specified, processes all available symbols.'
    )

    parser.add_argument(
        '--dly-folder',
        type=str,
        default=DLY_DATA_PATH,
        help=f'Path to DLY files folder (default: {DLY_DATA_PATH})'
    )

    parser.add_argument(
        '--market-data-path',
        type=str,
        default=MARKET_DATA_PATH,
        help=f'Path to market data HDF5 file (default: {MARKET_DATA_PATH})'
    )

    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=6,
        help='Maximum number of concurrent operations (default: 6)'
    )

    parser.add_argument(
        '--retry-attempts',
        type=int,
        default=3,
        help='Number of retry attempts for failed operations (default: 3)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without making changes'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def discover_available_symbols(dly_folder: str) -> List[str]:
    """Discover all available symbols in the DLY folder."""
    try:
        updater = DLYFolderUpdater(folder=dly_folder)
        return updater.list_tickers()
    except Exception as e:
        print(f"Error discovering symbols: {e}")
        return []


def get_existing_curve_symbols(client: DataClient) -> List[str]:
    """Get symbols that already have curve data in the market store."""
    curve_symbols = []
    try:
        market_keys = client.list_market_data()
        for key in market_keys:
            if '/curve' in key:
                # Extract symbol from 'market/SYMBOL_F/curve' format
                parts = key.split('/')
                if len(parts) >= 2:
                    symbol = parts[1]
                    if symbol.endswith('_F'):
                        base_symbol = symbol[:-2]
                        curve_symbols.append(base_symbol)

        return sorted(set(curve_symbols))
    except Exception as e:
        print(f"Error getting existing curve symbols: {e}")
        return []


async def refresh_single_symbol(symbol: str, dly_folder: str, client: DataClient,
                               retry_attempts: int = 3, verbose: bool = False) -> Dict[str, Any]:
    """Refresh curve calculations for a single symbol asynchronously."""
    if verbose:
        print(f"  Processing {symbol}...")

    start_time = time.time()
    result = {
        'symbol': symbol,
        'success': False,
        'message': '',
        'processing_time': 0,
        'contracts_processed': 0,
        'curve_keys_written': [],
        'attempts_made': 0,
        'validation_passed': False
    }

    # Retry logic for transient errors
    for attempt in range(retry_attempts):
        result['attempts_made'] = attempt + 1

        try:
            # Run curve building in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            manager = await loop.run_in_executor(
                None,
                lambda: build_curve_for_symbol(symbol, dly_folder)
            )

            if manager is None:
                result['message'] = 'Failed to build curve from DLY files'
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(0.5)  # Brief delay before retry
                    continue
                break

            # Check if we have valid curve data
            if manager.curve is None or manager.curve.empty:
                result['message'] = 'No curve data generated'
                break

            # Validate the curve data before writing
            validation_result = validate_curve_data(manager)
            if not validation_result['valid']:
                result['message'] = f'Curve validation failed: {validation_result["reason"]}'
                break

            # Write to HDF5 with proper locking
            write_success = await write_curve_to_hdf5(manager, client, symbol, verbose)
            if not write_success['success']:
                result['message'] = write_success['error']
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(1.0)  # Longer delay for HDF5 issues
                    continue
                break

            # Post-write validation
            post_validation = await validate_written_data(client, symbol, manager)

            # Success!
            result['success'] = True
            result['validation_passed'] = post_validation['valid']
            result['contracts_processed'] = len(manager.curve.columns) if manager.curve is not None else 0
            result['message'] = f'Successfully refreshed {result["contracts_processed"]} contracts'

            if not post_validation['valid']:
                result['message'] += f' (Warning: {post_validation["warning"]})'

            # List the keys that were written
            result['curve_keys_written'] = [
                f"market/{symbol}_F/curve",
                f"market/{symbol}_F/dte",
                f"market/{symbol}_F/expiry",
                f"market/{symbol}_F/front",
                f"market/{symbol}_F/seq_curve",
                f"market/{symbol}_F/seq_labels",
                f"market/{symbol}_F/seq_dte",
                f"market/{symbol}_F/seq_spreads"
            ]

            # Add volume/OI keys if available
            if manager.curve_volume is not None:
                result['curve_keys_written'].extend([
                    f"market/{symbol}_F/curve_volume",
                    f"market/{symbol}_F/seq_volume"
                ])

            if manager.curve_oi is not None:
                result['curve_keys_written'].extend([
                    f"market/{symbol}_F/curve_oi",
                    f"market/{symbol}_F/seq_oi"
                ])

            break  # Success, exit retry loop

        except Exception as e:
            error_msg = f'Error processing {symbol} (attempt {attempt + 1}): {str(e)}'
            result['message'] = error_msg

            if attempt < retry_attempts - 1:
                if verbose:
                    print(f"    Retrying {symbol} after error: {str(e)}")
                await asyncio.sleep(1.0)  # Wait before retry
            else:
                # Final attempt failed
                if verbose:
                    print(f"    All attempts failed for {symbol}")

    result['processing_time'] = time.time() - start_time
    return result


def build_curve_for_symbol(symbol: str, dly_folder: str) -> Optional[DLYContractManager]:
    """Build curve for a symbol synchronously (called in executor)."""
    try:
        manager = DLYContractManager(ticker=symbol, folder=dly_folder)
        curve_result = manager.run(save=False)

        if curve_result is None:
            return None

        return manager
    except Exception:
        return None


def validate_curve_data(manager: DLYContractManager) -> Dict[str, Any]:
    """Validate curve data before writing."""
    try:
        if manager.curve is None or manager.curve.empty:
            return {'valid': False, 'reason': 'Empty curve data'}

        if manager.seq_prices is None:
            return {'valid': False, 'reason': 'Missing sequential prices'}

        if manager.seq_dte is None:
            return {'valid': False, 'reason': 'Missing sequential DTE data'}

        # Check for reasonable DTE values
        if hasattr(manager.seq_dte, 'values'):
            dte_values = manager.seq_dte.values
            valid_dte = dte_values[~pd.isna(dte_values)]
            if len(valid_dte) > 0:
                max_dte = valid_dte.max()
                if max_dte < 30:  # Should have contracts extending at least 30 days
                    return {'valid': False, 'reason': f'Insufficient contract coverage (max DTE: {max_dte})'}

        return {'valid': True, 'reason': 'Validation passed'}

    except Exception as e:
        return {'valid': False, 'reason': f'Validation error: {str(e)}'}


async def write_curve_to_hdf5(manager: DLYContractManager, client: DataClient,
                            symbol: str, verbose: bool = False) -> Dict[str, Any]:
    """Write curve data to HDF5 with proper locking."""
    try:
        # Use global HDF5 lock to prevent conflicts
        async with HDF5_LOCK:
            if verbose:
                print(f"    Writing {symbol} to HDF5...")

            # Set HDF path and save using thread pool
            manager.hdf_path = client.market_data_path

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, manager.save_hdf)

            return {'success': True, 'error': None}

    except Exception as e:
        return {'success': False, 'error': f'HDF5 write error: {str(e)}'}


async def validate_written_data(client: DataClient, symbol: str,
                               original_manager: DLYContractManager) -> Dict[str, Any]:
    """Validate that data was written correctly to HDF5."""
    try:
        # Check if we can read back the curve data
        curve_key = f"market/{symbol}_F/curve"

        loop = asyncio.get_event_loop()
        written_data = await loop.run_in_executor(
            None,
            lambda: client.get_market_tail(curve_key, 100)
        )

        if written_data.empty:
            return {'valid': False, 'warning': 'Could not read back written curve data'}

        # Basic sanity checks
        expected_contracts = len(original_manager.curve.columns) if original_manager.curve is not None else 0
        actual_contracts = len(written_data.columns)

        if actual_contracts < expected_contracts * 0.8:  # Allow some tolerance
            return {
                'valid': False,
                'warning': f'Contract count mismatch: expected ~{expected_contracts}, got {actual_contracts}'
            }

        return {'valid': True, 'warning': None}

    except Exception as e:
        return {'valid': False, 'warning': f'Validation read error: {str(e)}'}


async def refresh_curves_batch(symbols: List[str], dly_folder: str, client: DataClient,
                              max_concurrent: int = 6, retry_attempts: int = 3,
                              dry_run: bool = False, verbose: bool = False) -> Dict[str, Any]:
    """Refresh curve calculations for multiple symbols asynchronously."""

    print(f"\n{'='*60}")
    print(f"REFRESHING CURVE CALCULATIONS (ASYNC)")
    print(f"{'='*60}")
    print(f"Symbols to process: {len(symbols)}")
    print(f"DLY folder: {dly_folder}")
    print(f"Market data: {client.market_data_path}")
    print(f"Max concurrent: {max_concurrent}")
    print(f"Retry attempts: {retry_attempts}")
    print(f"Dry run: {dry_run}")
    print(f"{'='*60}\n")

    if dry_run:
        print("DRY RUN MODE - No changes will be made\n")
        for symbol in symbols:
            print(f"  Would process: {symbol}")
        return {
            'total_symbols': len(symbols),
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'results': [],
            'dry_run': True
        }

    # Create semaphore to limit concurrent operations
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(symbol: str) -> Dict[str, Any]:
        """Process a single symbol with concurrency control."""
        async with semaphore:
            return await refresh_single_symbol(
                symbol, dly_folder, client, retry_attempts, verbose
            )

    # Progress tracking
    completed = 0
    results = []
    successful = 0
    failed = 0

    print(f"Processing {len(symbols)} symbols concurrently (max {max_concurrent})...\n")

    # Create tasks for all symbols
    tasks = [process_with_semaphore(symbol) for symbol in symbols]

    # Process with progress updates
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        completed += 1

        symbol = result['symbol']
        if result['success']:
            successful += 1
            status_char = "✓"
            validation_note = ""
            if not result.get('validation_passed', True):
                validation_note = " (validation warning)"
            print(f"[{completed}/{len(symbols)}] {status_char} {symbol}: {result['message']} ({result['processing_time']:.1f}s){validation_note}")

            if verbose and result.get('attempts_made', 1) > 1:
                print(f"    Completed after {result['attempts_made']} attempts")
            if verbose:
                print(f"    Keys written: {len(result['curve_keys_written'])}")
        else:
            failed += 1
            attempts_note = ""
            if result.get('attempts_made', 1) > 1:
                attempts_note = f" ({result['attempts_made']} attempts)"
            print(f"[{completed}/{len(symbols)}] ✗ {symbol}: {result['message']} ({result['processing_time']:.1f}s){attempts_note}")

    # Sort results by symbol name for consistent output
    results.sort(key=lambda r: r['symbol'])

    summary = {
        'total_symbols': len(symbols),
        'processed': len(results),
        'successful': successful,
        'failed': failed,
        'results': results,
        'dry_run': False
    }

    return summary


def print_summary(summary: Dict[str, Any]):
    """Print a summary of the refresh operation."""
    print(f"\n{'='*60}")
    print(f"REFRESH SUMMARY")
    print(f"{'='*60}")

    if summary['dry_run']:
        print(f"Mode: DRY RUN (no changes made)")
    else:
        print(f"Mode: LIVE RUN")

    print(f"Total symbols: {summary['total_symbols']}")
    print(f"Processed: {summary['processed']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")

    if summary['successful'] > 0:
        success_rate = (summary['successful'] / summary['total_symbols']) * 100
        print(f"Success rate: {success_rate:.1f}%")

    # Show failed symbols if any
    if summary['failed'] > 0:
        print(f"\nFailed symbols:")
        for result in summary['results']:
            if not result['success']:
                print(f"  - {result['symbol']}: {result['message']}")

    # Show processing times and retry statistics
    if not summary['dry_run'] and summary['results']:
        total_time = sum(r['processing_time'] for r in summary['results'])
        avg_time = total_time / len(summary['results'])

        # Calculate retry statistics
        total_attempts = sum(r.get('attempts_made', 1) for r in summary['results'])
        symbols_with_retries = sum(1 for r in summary['results'] if r.get('attempts_made', 1) > 1)

        # Validation statistics
        validation_warnings = sum(1 for r in summary['results']
                                if r['success'] and not r.get('validation_passed', True))

        print(f"\nTiming (Async Processing):")
        print(f"  Total processing time: {total_time:.1f}s")
        print(f"  Average per symbol: {avg_time:.1f}s")
        print(f"  Total attempts: {total_attempts}")
        if symbols_with_retries > 0:
            print(f"  Symbols requiring retries: {symbols_with_retries}")
        if validation_warnings > 0:
            print(f"  Validation warnings: {validation_warnings}")

        # Show fastest and slowest
        if len(summary['results']) > 1:
            times = [r['processing_time'] for r in summary['results']]
            fastest = min(times)
            slowest = max(times)
            print(f"  Fastest symbol: {fastest:.1f}s")
            print(f"  Slowest symbol: {slowest:.1f}s")

    print(f"{'='*60}")


async def main():
    """Main function."""
    args = parse_arguments()

    # Validate DLY folder
    dly_folder = Path(args.dly_folder)
    if not dly_folder.exists():
        print(f"Error: DLY folder does not exist: {dly_folder}")
        sys.exit(1)

    # Initialize data client
    try:
        client = DataClient(market_path=args.market_data_path)
    except Exception as e:
        print(f"Error initializing DataClient: {e}")
        sys.exit(1)

    # Determine symbols to process
    if args.symbols:
        # Use specified symbols
        requested_symbols = [s.strip().upper() for s in args.symbols.split(',')]
        print(f"Requested symbols: {requested_symbols}")

        # Verify symbols exist in DLY folder
        available_symbols = discover_available_symbols(str(dly_folder))
        symbols_to_process = []

        for symbol in requested_symbols:
            if symbol in available_symbols:
                symbols_to_process.append(symbol)
            else:
                print(f"Warning: Symbol {symbol} not found in DLY folder")

        if not symbols_to_process:
            print("Error: No valid symbols found to process")
            sys.exit(1)

    else:
        # Auto-discover symbols that have existing curve data
        print("Auto-discovering symbols with existing curve data...")
        existing_symbols = get_existing_curve_symbols(client)
        available_symbols = discover_available_symbols(str(dly_folder))

        # Only process symbols that have both existing curve data and DLY files
        symbols_to_process = [s for s in existing_symbols if s in available_symbols]

        if not symbols_to_process:
            print("No symbols found with both existing curve data and DLY files")
            sys.exit(1)

        print(f"Found {len(symbols_to_process)} symbols to refresh: {symbols_to_process}")

    # Refresh curves
    try:
        summary = await refresh_curves_batch(
            symbols=symbols_to_process,
            dly_folder=str(dly_folder),
            client=client,
            max_concurrent=args.max_concurrent,
            retry_attempts=args.retry_attempts,
            dry_run=args.dry_run,
            verbose=args.verbose
        )

        print_summary(summary)

        # Exit with appropriate code
        if summary['failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error during refresh operation: {e}")
        sys.exit(1)


def run_main():
    """Entry point that handles asyncio setup."""
    try:
        # Use asyncio.run for Python 3.7+
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(130)


if __name__ == '__main__':
    run_main()