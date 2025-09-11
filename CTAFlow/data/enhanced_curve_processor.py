#!/usr/bin/env python3
"""
Enhanced Curve Processor: Batch processing with enhanced roll tracking for specific tickers.

This module provides selective futures curve processing with enhanced roll tracking
for a user-specified list of ticker symbols.
"""

import time
import concurrent.futures
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

from CTAFlow.data.contract_handling.roll_date_manager import create_enhanced_curve_manager_with_roll_tracking


def process_enhanced_curves_for_ticker_list(
    ticker_list: List[str],
    prefer_front_series: bool = True,
    match_tol: float = 0.01,
    rel_jump_thresh: float = 0.01,
    robust_k: float = 4.0,
    near_expiry_days: int = 7,
    min_persistence_days: int = 3,
    validate_rolls: bool = True,
    track_rolls: bool = True,
    max_workers: Optional[int] = None,
    progress: bool = True,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Process futures curves with enhanced roll tracking for a specific list of tickers.
    
    This function creates enhanced curve managers with roll tracking for each ticker
    in the provided list, allowing selective processing instead of all available data.
    
    Parameters:
    -----------
    ticker_list : List[str]
        List of ticker symbols to process (e.g., ['CL_F', 'ZC_F', 'ES_F'])
    prefer_front_series : bool, default True
        Whether to prefer front month series in curve construction
    match_tol : float, default 0.01
        Price matching tolerance for curve alignment
    rel_jump_thresh : float, default 0.01
        Relative jump threshold for roll detection
    robust_k : float, default 4.0
        Robust scaling parameter for outlier detection
    near_expiry_days : int, default 7
        Conservative roll timing - days before expiry to consider rolling
    min_persistence_days : int, default 3
        Minimum days a roll signal must persist before executing
    validate_rolls : bool, default True
        Whether to validate roll timing using volume/OI criteria
    track_rolls : bool, default True
        Whether to track and persist roll events to disk
    max_workers : int, optional
        Maximum number of worker threads for parallel processing.
        If None, uses sequential processing for better control.
    progress : bool, default True
        Whether to show progress messages during processing
    debug : bool, default False
        Whether to show detailed debugging information
        
    Returns:
    --------
    Dict[str, Any]
        Processing results containing:
        - 'successful': Dict mapping ticker -> success message
        - 'failed': Dict mapping ticker -> error message  
        - 'summary': Dict with processing statistics
        - 'roll_tracking': Dict mapping ticker -> roll tracking info
        
    Examples:
    ---------
    >>> # Process specific energy tickers
    >>> energy_tickers = ['CL_F', 'NG_F', 'HO_F', 'RB_F']
    >>> results = process_enhanced_curves_for_ticker_list(
    ...     ticker_list=energy_tickers,
    ...     progress=True,
    ...     debug=False
    ... )
    >>> print(f"Processed {len(results['successful'])} tickers successfully")
    
    >>> # Process agricultural tickers with custom parameters
    >>> ag_tickers = ['ZC_F', 'ZS_F', 'ZW_F']
    >>> results = process_enhanced_curves_for_ticker_list(
    ...     ticker_list=ag_tickers,
    ...     near_expiry_days=10,  # More conservative roll timing
    ...     min_persistence_days=5,
    ...     max_workers=3  # Parallel processing
    ... )
    
    >>> # Check roll tracking results
    >>> for ticker, roll_info in results['roll_tracking'].items():
    ...     print(f"{ticker}: {roll_info['roll_count']} rolls tracked")
    """
    
    if not ticker_list:
        return {
            'successful': {},
            'failed': {},
            'summary': {'error': 'Empty ticker list provided'},
            'roll_tracking': {}
        }
    
    # Validate ticker format
    valid_tickers = []
    for ticker in ticker_list:
        ticker = ticker.upper().strip()
        if not ticker.endswith('_F'):
            if progress:
                print(f"Warning: Adding '_F' suffix to ticker '{ticker}'")
            ticker = f"{ticker}_F"
        valid_tickers.append(ticker)
    
    if progress:
        print(f"Processing enhanced curves for {len(valid_tickers)} specific tickers:")
        for ticker in valid_tickers:
            print(f"  {ticker}")
        print()
    
    # Initialize results
    results = {
        'successful': {},
        'failed': {},
        'summary': {},
        'roll_tracking': {}
    }
    
    start_time = time.time()
    
    def process_single_ticker_enhanced(ticker_symbol: str) -> Tuple[str, bool, str, Dict]:
        """
        Worker function to process enhanced curves for a single ticker.
        
        Returns:
        --------
        Tuple[str, bool, str, Dict]
            (ticker_symbol, success, message, roll_tracking_info)
        """
        try:
            # Create enhanced curve manager with roll tracking
            curve_manager = create_enhanced_curve_manager_with_roll_tracking(ticker_symbol)
            
            # Run enhanced curve processing with roll tracking
            result = curve_manager.run(
                prefer_front_series=prefer_front_series,
                match_tol=match_tol,
                rel_jump_thresh=rel_jump_thresh,
                robust_k=robust_k,
                near_expiry_days=near_expiry_days,
                min_persistence_days=min_persistence_days,
                validate_rolls=validate_rolls,
                track_rolls=track_rolls,
                debug=debug
            )
            
            # Collect roll tracking information
            roll_tracking_info = {
                'roll_count': 0,
                'roll_dataframe_available': False,
                'validation_summary': None
            }
            
            # Check if roll data was created and tracked
            if hasattr(curve_manager, 'roll_dataframe') and curve_manager.roll_dataframe is not None:
                roll_tracking_info['roll_count'] = len(curve_manager.roll_dataframe)
                roll_tracking_info['roll_dataframe_available'] = True
                
                # Create summary of roll validation results
                if 'validation_result' in curve_manager.roll_dataframe.columns:
                    validation_counts = curve_manager.roll_dataframe['validation_result'].value_counts()
                    roll_tracking_info['validation_summary'] = validation_counts.to_dict()
            
            # Check if roll manager has additional information
            if hasattr(curve_manager, 'roll_manager'):
                roll_history_count = len(curve_manager.roll_manager.roll_history)
                roll_tracking_info['historical_rolls'] = roll_history_count
            
            # Success message with roll tracking info
            roll_info_str = ""
            if roll_tracking_info['roll_count'] > 0:
                roll_info_str = f" ({roll_tracking_info['roll_count']} rolls tracked)"
            
            success_message = f"Successfully processed enhanced curves for {ticker_symbol}{roll_info_str}"
            
            return (ticker_symbol, True, success_message, roll_tracking_info)
            
        except Exception as e:
            error_message = f"Failed to process enhanced curves for {ticker_symbol}: {str(e)}"
            empty_tracking = {'roll_count': 0, 'roll_dataframe_available': False, 'error': str(e)}
            return (ticker_symbol, False, error_message, empty_tracking)
    
    # Process tickers (with optional multi-threading)
    processed_count = 0
    
    if max_workers and max_workers > 1:
        # Multi-threaded processing
        if progress:
            print(f"Using {max_workers} threads for parallel enhanced processing...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(process_single_ticker_enhanced, ticker): ticker 
                for ticker in valid_tickers
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker_symbol, success, message, roll_info = future.result()
                processed_count += 1
                
                if success:
                    results['successful'][ticker_symbol] = message
                    results['roll_tracking'][ticker_symbol] = roll_info
                    if progress:
                        print(f"✓ [{processed_count}/{len(valid_tickers)}] {message}")
                else:
                    results['failed'][ticker_symbol] = message
                    results['roll_tracking'][ticker_symbol] = roll_info
                    if progress:
                        print(f"✗ [{processed_count}/{len(valid_tickers)}] {message}")
    else:
        # Sequential processing (default for better control and debugging)
        if progress:
            print("Processing sequentially with enhanced roll tracking...")
        
        for ticker in valid_tickers:
            ticker_symbol, success, message, roll_info = process_single_ticker_enhanced(ticker)
            processed_count += 1
            
            if success:
                results['successful'][ticker_symbol] = message
                results['roll_tracking'][ticker_symbol] = roll_info
                if progress:
                    print(f"✓ [{processed_count}/{len(valid_tickers)}] {message}")
            else:
                results['failed'][ticker_symbol] = message
                results['roll_tracking'][ticker_symbol] = roll_info
                if progress:
                    print(f"✗ [{processed_count}/{len(valid_tickers)}] {message}")
    
    # Generate summary
    total_time = time.time() - start_time
    successful_count = len(results['successful'])
    failed_count = len(results['failed'])
    total_rolls_tracked = sum(info.get('roll_count', 0) for info in results['roll_tracking'].values())
    
    results['summary'] = {
        'total_tickers': len(valid_tickers),
        'successful_count': successful_count,
        'failed_count': failed_count,
        'processing_time': total_time,
        'success_rate': (successful_count / len(valid_tickers)) * 100 if valid_tickers else 0,
        'total_rolls_tracked': total_rolls_tracked,
        'average_rolls_per_ticker': total_rolls_tracked / successful_count if successful_count > 0 else 0
    }
    
    if progress:
        print(f"\n" + "="*60)
        print("ENHANCED CURVE PROCESSING SUMMARY")
        print("="*60)
        print(f"Total tickers processed: {len(valid_tickers)}")
        print(f"Successfully processed: {successful_count}")
        print(f"Failed to process: {failed_count}")
        print(f"Success rate: {results['summary']['success_rate']:.1f}%")
        print(f"Total processing time: {total_time:.1f} seconds")
        print(f"Total rolls tracked: {total_rolls_tracked}")
        if successful_count > 0:
            print(f"Average time per ticker: {total_time / len(valid_tickers):.2f} seconds")
            print(f"Average rolls per ticker: {results['summary']['average_rolls_per_ticker']:.1f}")
        
        if results['failed']:
            print(f"\nFailed tickers:")
            for ticker, error in results['failed'].items():
                print(f"  {ticker}: {error}")
        
        if results['roll_tracking']:
            print(f"\nRoll tracking details:")
            for ticker, roll_info in results['roll_tracking'].items():
                if ticker in results['successful']:
                    roll_count = roll_info.get('roll_count', 0)
                    validation_info = ""
                    if roll_info.get('validation_summary'):
                        validation_info = f" (validation: {roll_info['validation_summary']})"
                    print(f"  {ticker}: {roll_count} rolls{validation_info}")
        
        print("="*60)
    
    return results


def process_energy_tickers_enhanced(
    progress: bool = True,
    max_workers: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to process major energy futures with enhanced roll tracking.
    
    Parameters:
    -----------
    progress : bool, default True
        Whether to show progress messages
    max_workers : int, optional
        Maximum number of worker threads
    **kwargs : dict
        Additional parameters passed to process_enhanced_curves_for_ticker_list
        
    Returns:
    --------
    Dict[str, Any]
        Processing results
    """
    energy_tickers = ['CL_F', 'NG_F', 'HO_F', 'RB_F', 'BZ_F']  # Major energy futures
    
    return process_enhanced_curves_for_ticker_list(
        ticker_list=energy_tickers,
        progress=progress,
        max_workers=max_workers,
        **kwargs
    )


def process_agricultural_tickers_enhanced(
    progress: bool = True,
    max_workers: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to process major agricultural futures with enhanced roll tracking.
    
    Parameters:
    -----------
    progress : bool, default True
        Whether to show progress messages
    max_workers : int, optional
        Maximum number of worker threads
    **kwargs : dict
        Additional parameters passed to process_enhanced_curves_for_ticker_list
        
    Returns:
    --------
    Dict[str, Any]
        Processing results
    """
    ag_tickers = ['ZC_F', 'ZS_F', 'ZW_F', 'CT_F', 'KC_F', 'SB_F', 'CC_F']  # Major ag futures
    
    return process_enhanced_curves_for_ticker_list(
        ticker_list=ag_tickers,
        progress=progress,
        max_workers=max_workers,
        **kwargs
    )


def process_metals_tickers_enhanced(
    progress: bool = True,
    max_workers: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to process major metals futures with enhanced roll tracking.
    
    Parameters:
    -----------
    progress : bool, default True
        Whether to show progress messages
    max_workers : int, optional
        Maximum number of worker threads
    **kwargs : dict
        Additional parameters passed to process_enhanced_curves_for_ticker_list
        
    Returns:
    --------
    Dict[str, Any]
        Processing results
    """
    metals_tickers = ['GC_F', 'SI_F', 'PL_F', 'PA_F', 'HG_F']  # Major metals futures
    
    return process_enhanced_curves_for_ticker_list(
        ticker_list=metals_tickers,
        progress=progress,
        max_workers=max_workers,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    print("Enhanced Curve Processor")
    print("Usage examples:")
    print()
    print("# Process specific tickers:")
    print("results = process_enhanced_curves_for_ticker_list(['CL_F', 'NG_F'])")
    print()
    print("# Process energy sector:")
    print("results = process_energy_tickers_enhanced()")
    print()
    print("# Process agricultural sector:")
    print("results = process_agricultural_tickers_enhanced()")
    print()
    print("# Process metals sector:")
    print("results = process_metals_tickers_enhanced()")