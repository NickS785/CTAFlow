"""Backtest reporting utilities for generating structured reports from backtest results.

This module provides tools for creating comprehensive reports from ScreenerBacktester
outputs, organized by ticker with top performers highlighted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence
import pandas as pd
import numpy as np
from .backtester import BacktestSummary, ScreenerBacktester


@dataclass
class TickerReport:
    """Report for a single ticker containing its best-performing patterns.

    Attributes:
        ticker: Ticker symbol
        top_patterns: List of (pattern_name, summary, score) tuples sorted by performance
        total_patterns: Total number of patterns tested for this ticker
        avg_return: Average total return across all patterns
        avg_sharpe: Average Sharpe ratio across all patterns
        best_pattern: Name of the best-performing pattern
        best_score: Performance score of the best pattern
    """
    ticker: str
    top_patterns: List[tuple[str, BacktestSummary, float]] = field(default_factory=list)
    total_patterns: int = 0
    avg_return: float = 0.0
    avg_sharpe: float = 0.0
    best_pattern: Optional[str] = None
    best_score: float = float('-inf')

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            'ticker': self.ticker,
            'total_patterns': self.total_patterns,
            'avg_return': self.avg_return,
            'avg_sharpe': self.avg_sharpe,
            'best_pattern': self.best_pattern,
            'best_score': self.best_score,
            'top_patterns': [
                {
                    'pattern': name,
                    'total_return': summary.total_return,
                    'mean_return': summary.mean_return,
                    'sharpe': summary.sharpe,
                    'hit_rate': summary.hit_rate,
                    'max_drawdown': summary.max_drawdown,
                    'trades': summary.trades,
                    'score': score,
                }
                for name, summary, score in self.top_patterns
            ]
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert top patterns to DataFrame with parsed pattern components."""
        if not self.top_patterns:
            return pd.DataFrame()

        rows = []
        for name, summary, score in self.top_patterns:
            # Parse pattern name: "TICKER|screen|pattern_details..."
            # Examples:
            #   "CL|usa_spring|month_start_1d"
            #   "NG|momentum_generic|momentum_so|st_momentum|session_0"
            #   "PL|usa_winter|weekday_mean|Wednesday"
            parts = name.split('|')

            ticker = parts[0] if len(parts) > 0 else self.ticker
            screen_name = parts[1] if len(parts) > 1 else ''

            # Extract pattern_name and pattern_type
            # For calendar patterns: "month_start_1d" -> pattern_name
            # For momentum patterns: "momentum_so|st_momentum|session_0" -> pattern_name
            # Pattern type is inferred from structure

            # Check for weekend_hedging first (can be in screen_name with only 2 parts)
            if 'weekend_hedging' in screen_name:
                pattern_type = 'weekend_hedging'
                pattern_name = screen_name
            elif len(parts) > 2:
                # Determine pattern type based on common keywords
                pattern_type = ''
                remaining_parts = parts[2:]

                # Check if it's a momentum pattern
                if any(p.startswith('momentum_') for p in remaining_parts):
                    pattern_type = 'momentum_generic'
                    pattern_name = '|'.join(remaining_parts)
                # Check if it's a calendar pattern
                elif any(keyword in '|'.join(remaining_parts) for keyword in ['month_', 'quarter_', 'week']):
                    pattern_type = 'calendar'
                    pattern_name = '|'.join(remaining_parts)
                # Check if it's a weekday pattern
                elif 'weekday' in '|'.join(remaining_parts):
                    pattern_type = remaining_parts[0] if remaining_parts else ''
                    pattern_name = '|'.join(remaining_parts[1:]) if len(remaining_parts) > 1 else ''
                # Check if it's a time predictive pattern
                elif 'time_predictive' in '|'.join(remaining_parts):
                    pattern_type = remaining_parts[0] if remaining_parts else ''
                    pattern_name = '|'.join(remaining_parts[1:]) if len(remaining_parts) > 1 else ''
                # Check if it's an orderflow pattern
                elif 'orderflow' in '|'.join(remaining_parts):
                    pattern_type = remaining_parts[0] if remaining_parts else ''
                    pattern_name = '|'.join(remaining_parts[1:]) if len(remaining_parts) > 1 else ''
                else:
                    # Generic: first part is pattern type, rest is pattern name
                    pattern_type = remaining_parts[0] if remaining_parts else ''
                    pattern_name = '|'.join(remaining_parts[1:]) if len(remaining_parts) > 1 else ''
            else:
                pattern_type = ''
                pattern_name = name

            row = {
                'ticker': ticker,
                'screen_name': screen_name,
                'pattern_name': pattern_name,
                'pattern_type': pattern_type,
                'total_return': summary.total_return,
                'mean_return': summary.mean_return,
                'sharpe': summary.sharpe,
                'hit_rate': summary.hit_rate,
                'max_drawdown': summary.max_drawdown,
                'trades': summary.trades,
                'score': score,
            }
            rows.append(row)

        return pd.DataFrame(rows)


@dataclass
class BacktestReport:
    """Comprehensive report aggregating backtest results across tickers.

    Attributes:
        ticker_reports: Dictionary mapping ticker symbols to TickerReport objects
        global_summary: Overall statistics across all tickers
        report_timestamp: When the report was generated
    """
    ticker_reports: Dict[str, TickerReport] = field(default_factory=dict)
    global_summary: Dict[str, Any] = field(default_factory=dict)
    report_timestamp: Optional[pd.Timestamp] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert full report to nested dictionary."""
        return {
            'timestamp': str(self.report_timestamp) if self.report_timestamp else None,
            'global_summary': self.global_summary,
            'ticker_reports': {
                ticker: report.to_dict()
                for ticker, report in self.ticker_reports.items()
            }
        }

    def to_dataframe(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Convert all ticker reports to a single DataFrame.

        Args:
            top_n: If specified, only include top N patterns per ticker

        Returns:
            DataFrame with all patterns from all tickers
        """
        frames = []
        for ticker, report in self.ticker_reports.items():
            df = report.to_dataframe()
            if not df.empty:
                if top_n is not None:
                    df = df.head(top_n)
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def get_top_patterns_overall(self, n: int = 10) -> pd.DataFrame:
        """Get the top N patterns across all tickers by score.

        Args:
            n: Number of top patterns to return

        Returns:
            DataFrame with top N patterns sorted by score
        """
        df = self.to_dataframe()
        if df.empty:
            return df

        return df.nlargest(n, 'score').reset_index(drop=True)

    def get_ticker_summary(self) -> pd.DataFrame:
        """Get summary statistics for each ticker.

        Returns:
            DataFrame with one row per ticker showing aggregate metrics
        """
        if not self.ticker_reports:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'ticker': ticker,
                'total_patterns': report.total_patterns,
                'avg_return': report.avg_return,
                'avg_sharpe': report.avg_sharpe,
                'best_pattern': report.best_pattern,
                'best_score': report.best_score,
            }
            for ticker, report in self.ticker_reports.items()
        ]).sort_values('best_score', ascending=False).reset_index(drop=True)


class BacktestReportGenerator:
    """Generator for creating structured backtest reports.

    This class processes backtest results from ScreenerBacktester and generates
    reports organized by ticker with top performers highlighted.

    Args:
        scoring_fn: Optional custom scoring function. Defaults to return/drawdown ratio.
        top_n: Number of top patterns to include per ticker (default: 10)
    """

    def __init__(
        self,
        scoring_fn: Optional[callable] = None,
        top_n: int = 10,
    ):
        self.scoring_fn = scoring_fn or ScreenerBacktester.ranking_score
        self.top_n = top_n

    def generate_report_from_ticker_dict(
        self,
        ticker_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
        *,
        include_global_summary: bool = True,
    ) -> BacktestReport:
        """Generate a backtest report from nested ticker→pattern→results structure.

        This method handles results from calling concurrent_pattern_backtests()
        separately for each ticker, producing a structure like:
        {
            'CL': {'pattern1': {...}, 'pattern2': {...}},
            'GC': {'pattern1': {...}, 'pattern2': {...}},
        }

        Args:
            ticker_results: Nested dict {ticker: {pattern: result}}
            include_global_summary: Whether to compute global statistics

        Returns:
            BacktestReport object containing organized results

        Example:
            >>> screener_tgts = {}
            >>> for ticker, data in hs.data.items():
            ...     screener_tgts[ticker] = sp.concurrent_pattern_backtests(
            ...         data, patterns, verbose=True
            ...     )
            >>> generator = BacktestReportGenerator(top_n=5)
            >>> report = generator.generate_report_from_ticker_dict(screener_tgts)
        """
        ticker_reports = {}

        for ticker, pattern_results in ticker_results.items():
            patterns_list = []

            for pattern_name, result in pattern_results.items():
                # Extract summary
                summary = result.get('summary')
                if not isinstance(summary, BacktestSummary):
                    continue

                # Calculate score
                score = self.scoring_fn(summary)
                if not np.isfinite(score):
                    score = float('-inf')

                patterns_list.append((pattern_name, summary, score))

            if not patterns_list:
                continue

            # Sort by score descending
            patterns_list.sort(key=lambda x: x[2], reverse=True)

            # Calculate statistics
            total_patterns = len(patterns_list)
            returns = [p[1].total_return for p in patterns_list]
            sharpes = [p[1].sharpe for p in patterns_list if np.isfinite(p[1].sharpe)]

            avg_return = float(np.mean(returns)) if returns else 0.0
            avg_sharpe = float(np.mean(sharpes)) if sharpes else np.nan

            best_pattern = patterns_list[0][0] if patterns_list else None
            best_score = patterns_list[0][2] if patterns_list else float('-inf')

            # Create ticker report
            ticker_reports[ticker] = TickerReport(
                ticker=ticker,
                top_patterns=patterns_list[:self.top_n],
                total_patterns=total_patterns,
                avg_return=avg_return,
                avg_sharpe=avg_sharpe,
                best_pattern=best_pattern,
                best_score=best_score,
            )

        # Global summary
        global_summary = {}
        if include_global_summary:
            global_summary = self._compute_global_summary(ticker_reports)

        return BacktestReport(
            ticker_reports=ticker_reports,
            global_summary=global_summary,
            report_timestamp=pd.Timestamp.now(),
        )

    def generate_report(
        self,
        results: Mapping[str, Mapping[str, Any]],
        *,
        group_field: str = 'ticker',
        include_global_summary: bool = True,
    ) -> BacktestReport:
        """Generate a comprehensive backtest report from raw results.

        Args:
            results: Dictionary mapping pattern names to backtest result dicts
                    (output from ScreenerBacktester.batch_patterns or similar)
            group_field: Field name used for grouping (typically 'ticker')
            include_global_summary: Whether to compute global statistics

        Returns:
            BacktestReport object containing organized results

        Example:
            >>> backtester = ScreenerBacktester(use_gpu=True)
            >>> results = backtester.batch_patterns(xy_map, group_field='ticker')
            >>> generator = BacktestReportGenerator(top_n=5)
            >>> report = generator.generate_report(results)
            >>> ticker_df = report.get_ticker_summary()
            >>> top_patterns = report.get_top_patterns_overall(n=10)
        """
        # Organize results by ticker
        ticker_data: Dict[str, List[tuple[str, BacktestSummary, float]]] = {}

        for pattern_name, result in results.items():
            # Extract summary
            summary = result.get('summary')
            if not isinstance(summary, BacktestSummary):
                continue

            # Calculate score
            score = self.scoring_fn(summary)
            if not np.isfinite(score):
                score = float('-inf')

            # Check for group breakdown (per-ticker results)
            if 'group_breakdown' in result and result.get('group_field') == group_field:
                breakdown = result['group_breakdown']
                for ticker, metrics in breakdown.items():
                    # Create BacktestSummary for this ticker
                    ticker_summary = BacktestSummary(
                        total_return=metrics.get('total_return', 0.0),
                        mean_return=metrics.get('mean_return', 0.0),
                        hit_rate=np.nan,  # Not available in group breakdown
                        sharpe=np.nan,
                        max_drawdown=0.0,  # Not available in group breakdown
                        trades=metrics.get('trades', 0),
                    )
                    ticker_score = self._score_ticker_breakdown(ticker_summary, metrics)

                    if ticker not in ticker_data:
                        ticker_data[ticker] = []
                    ticker_data[ticker].append((pattern_name, ticker_summary, ticker_score))
            else:
                # No breakdown - extract ticker from pattern name
                # Pattern name format: "TICKER|screen|pattern|details..."
                parts = pattern_name.split('|')
                if len(parts) >= 1 and parts[0] and parts[0] != '__overall__':
                    ticker = parts[0]
                else:
                    ticker = '__overall__'

                if ticker not in ticker_data:
                    ticker_data[ticker] = []
                ticker_data[ticker].append((pattern_name, summary, score))

        # Build ticker reports
        ticker_reports = {}
        for ticker, patterns in ticker_data.items():
            # Sort by score descending
            patterns.sort(key=lambda x: x[2], reverse=True)

            # Calculate statistics
            total_patterns = len(patterns)
            returns = [p[1].total_return for p in patterns]
            sharpes = [p[1].sharpe for p in patterns if np.isfinite(p[1].sharpe)]

            avg_return = float(np.mean(returns)) if returns else 0.0
            avg_sharpe = float(np.mean(sharpes)) if sharpes else np.nan

            best_pattern = patterns[0][0] if patterns else None
            best_score = patterns[0][2] if patterns else float('-inf')

            # Create report
            report = TickerReport(
                ticker=ticker,
                top_patterns=patterns[:self.top_n],
                total_patterns=total_patterns,
                avg_return=avg_return,
                avg_sharpe=avg_sharpe,
                best_pattern=best_pattern,
                best_score=best_score,
            )
            ticker_reports[ticker] = report

        # Global summary
        global_summary = {}
        if include_global_summary:
            global_summary = self._compute_global_summary(ticker_reports)

        return BacktestReport(
            ticker_reports=ticker_reports,
            global_summary=global_summary,
            report_timestamp=pd.Timestamp.now(),
        )

    def _score_ticker_breakdown(
        self,
        summary: BacktestSummary,
        metrics: Dict[str, Any]
    ) -> float:
        """Score a ticker from group_breakdown metrics.

        Since group_breakdown doesn't include drawdown, use a simpler metric.
        """
        total_return = metrics.get('total_return', 0.0)
        trades = metrics.get('trades', 0)

        if trades == 0:
            return float('-inf')

        # Use return per trade as a simple metric
        return total_return / max(trades, 1)

    def _compute_global_summary(
        self,
        ticker_reports: Dict[str, TickerReport]
    ) -> Dict[str, Any]:
        """Compute aggregate statistics across all tickers."""
        if not ticker_reports:
            return {}

        total_tickers = len(ticker_reports)
        total_patterns = sum(r.total_patterns for r in ticker_reports.values())

        avg_returns = [r.avg_return for r in ticker_reports.values()]
        avg_sharpes = [r.avg_sharpe for r in ticker_reports.values() if np.isfinite(r.avg_sharpe)]
        best_scores = [r.best_score for r in ticker_reports.values() if np.isfinite(r.best_score)]

        return {
            'total_tickers': total_tickers,
            'total_patterns_tested': total_patterns,
            'avg_patterns_per_ticker': total_patterns / total_tickers if total_tickers > 0 else 0,
            'mean_avg_return': float(np.mean(avg_returns)) if avg_returns else 0.0,
            'mean_avg_sharpe': float(np.mean(avg_sharpes)) if avg_sharpes else np.nan,
            'best_overall_score': float(np.max(best_scores)) if best_scores else float('-inf'),
        }

    def generate_from_batch(
        self,
        backtester: ScreenerBacktester,
        xy_map: Mapping[str, pd.DataFrame],
        *,
        threshold: float = 0.0,
        use_side_hint: bool = True,
        group_field: str = 'ticker',
        **kwargs
    ) -> BacktestReport:
        """Convenience method to run batch backtest and generate report.

        Args:
            backtester: ScreenerBacktester instance
            xy_map: Dictionary mapping pattern names to XY DataFrames
            threshold: Signal threshold
            use_side_hint: Whether to use correlation hints
            group_field: Field to group by (typically 'ticker')
            **kwargs: Additional arguments passed to batch_patterns

        Returns:
            BacktestReport object

        Example:
            >>> backtester = ScreenerBacktester(use_gpu=True)
            >>> generator = BacktestReportGenerator(top_n=5)
            >>> report = generator.generate_from_batch(
            ...     backtester, xy_map, group_field='ticker'
            ... )
        """
        results = backtester.batch_patterns(
            xy_map,
            threshold=threshold,
            use_side_hint=use_side_hint,
            group_field=group_field,
            **kwargs
        )

        return self.generate_report(results, group_field=group_field)


def print_report_summary(report: BacktestReport, top_n: int = 5) -> None:
    """Pretty print a summary of the backtest report.

    Args:
        report: BacktestReport to summarize
        top_n: Number of top items to display per section
    """
    print("=" * 80)
    print("BACKTEST REPORT SUMMARY")
    print("=" * 80)

    # Global summary
    if report.global_summary:
        print("\nGlobal Statistics:")
        for key, value in report.global_summary.items():
            if isinstance(value, float):
                if np.isfinite(value):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")

    # Top tickers
    ticker_summary = report.get_ticker_summary()
    if not ticker_summary.empty:
        print(f"\nTop {min(top_n, len(ticker_summary))} Tickers by Best Score:")
        for idx, row in ticker_summary.head(top_n).iterrows():
            print(f"\n  {idx + 1}. {row['ticker']}")
            print(f"     Patterns tested: {row['total_patterns']}")
            print(f"     Avg return: {row['avg_return']:.4f}")
            if np.isfinite(row['avg_sharpe']):
                print(f"     Avg Sharpe: {row['avg_sharpe']:.4f}")
            print(f"     Best pattern: {row['best_pattern']}")
            print(f"     Best score: {row['best_score']:.4f}")

    # Top patterns overall
    top_patterns = report.get_top_patterns_overall(n=top_n)
    if not top_patterns.empty:
        print(f"\nTop {min(top_n, len(top_patterns))} Patterns Overall:")
        for idx, row in top_patterns.iterrows():
            print(f"\n  {idx + 1}. {row['pattern']} ({row['ticker']})")
            print(f"     Total return: {row['total_return']:.4f}")
            print(f"     Mean return: {row['mean_return']:.4f}")
            if np.isfinite(row['sharpe']):
                print(f"     Sharpe: {row['sharpe']:.4f}")
            if np.isfinite(row['hit_rate']):
                print(f"     Hit rate: {row['hit_rate']:.2%}")
            print(f"     Trades: {row['trades']}")
            print(f"     Score: {row['score']:.4f}")

    print("\n" + "=" * 80)