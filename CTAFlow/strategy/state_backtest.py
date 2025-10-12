#!/usr/bin/env python3
"""
Backtesting framework specifically designed for MultiStateStrategy implementations.

This module provides comprehensive backtesting capabilities for multi-state trading
strategies, including state transition analysis, performance metrics, and detailed
trade execution simulation.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings

from .intraday import MultiStateStrategy, StateResult, StateContext


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    action: str  # 'BUY' or 'SELL'
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_reason: str = "unknown"
    states_traversed: int = 0
    strategy_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateTransitionRecord:
    """Record of a state transition."""
    timestamp: datetime
    from_state: int
    to_state: int
    price: float
    volume: Optional[float] = None
    reason: str = ""
    context_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResults:
    """Complete backtest results and analytics."""

    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # State analysis
    state_statistics: Dict[int, Dict[str, Any]]
    avg_states_per_trade: float
    state_transition_matrix: np.ndarray

    # Detailed records
    trades: List[TradeRecord]
    state_transitions: List[StateTransitionRecord]
    equity_curve: pd.Series
    drawdown_curve: pd.Series

    # Strategy-specific metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class StateBacktester:
    """
    Comprehensive backtesting framework for MultiStateStrategy implementations.

    Features:
    - Tick-by-tick strategy execution simulation
    - State transition tracking and analysis
    - Performance metrics calculation
    - Trade execution with slippage and commission
    - Detailed reporting and visualization
    - Strategy comparison capabilities
    """

    def __init__(self,
                 initial_capital: float = 100000.0,
                 commission_per_trade: float = 5.0,
                 slippage_bps: float = 2.0,
                 position_sizing: str = 'fixed',  # 'fixed', 'percent_risk', 'kelly'
                 position_size: float = 1.0,
                 max_position_risk: float = 0.02,  # 2% risk per trade
                 enable_detailed_logging: bool = False):
        """
        Initialize the state backtester.

        Args:
            initial_capital: Starting capital for backtest
            commission_per_trade: Fixed commission per trade
            slippage_bps: Slippage in basis points
            position_sizing: Position sizing method
            position_size: Base position size (contracts or percentage)
            max_position_risk: Maximum risk per trade as percentage of capital
            enable_detailed_logging: Enable detailed state logging
        """
        self.initial_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.slippage_bps = slippage_bps
        self.position_sizing = position_sizing
        self.position_size = position_size
        self.max_position_risk = max_position_risk
        self.enable_detailed_logging = enable_detailed_logging

        # Backtest state
        self.current_capital = initial_capital
        self.current_position = 0.0
        self.current_position_price = 0.0
        self.trades: List[TradeRecord] = []
        self.state_transitions: List[StateTransitionRecord] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

        # Analytics
        self.state_counts: Dict[int, int] = {}
        self.state_durations: Dict[int, List[float]] = {}
        self.transition_matrix: Dict[Tuple[int, int], int] = {}

    def backtest_strategy(self,
                         strategy: MultiStateStrategy,
                         data: pd.DataFrame,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> BacktestResults:
        """
        Run backtest on a MultiStateStrategy.

        Args:
            strategy: MultiStateStrategy instance to test
            data: Market data DataFrame with OHLCV columns
            start_date: Start date for backtest (None = use all data)
            end_date: End date for backtest (None = use all data)

        Returns:
            BacktestResults object with comprehensive analysis
        """
        # Filter data by date range
        test_data = self._prepare_data(data, start_date, end_date)

        # Reset backtest state
        self._reset_backtest_state()

        # Track strategy state changes
        previous_state = strategy.get_current_state()
        state_entry_time = test_data.index[0] if len(test_data) > 0 else datetime.now()

        # Process each tick
        for timestamp, row in test_data.iterrows():
            # Convert row to Series with timestamp
            tick_data = row.copy()
            tick_data['timestamp'] = timestamp

            # Track state before processing
            pre_state = strategy.get_current_state()

            # Process tick through strategy
            trade_signal = strategy.process_tick(tick_data)

            # Track state after processing
            post_state = strategy.get_current_state()

            # Record state transition if changed
            if pre_state != post_state:
                self._record_state_transition(
                    timestamp, pre_state, post_state,
                    row['close'], row.get('volume', 0),
                    strategy.get_context_summary()
                )

                # Update state duration tracking
                if pre_state in self.state_durations:
                    duration = (timestamp - state_entry_time).total_seconds() / 60.0  # minutes
                    self.state_durations[pre_state].append(duration)
                else:
                    self.state_durations[pre_state] = []

                state_entry_time = timestamp

            # Update state count
            self.state_counts[post_state] = self.state_counts.get(post_state, 0) + 1

            # Execute trade if signal generated
            if trade_signal:
                self._execute_trade(trade_signal, tick_data, strategy.context)

            # Update equity curve
            portfolio_value = self._calculate_portfolio_value(row['close'])
            self.equity_curve.append((timestamp, portfolio_value))

        # Close any open position at end
        if self.current_position != 0:
            final_price = test_data.iloc[-1]['close']
            self._close_position(final_price, test_data.index[-1], "backtest_end")

        # Generate results
        return self._generate_results(test_data.index[0], test_data.index[-1])

    def _prepare_data(self,
                     data: pd.DataFrame,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Prepare and filter data for backtesting."""
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Add volume if missing
        if 'volume' not in data.columns:
            data = data.copy()
            data['volume'] = 1000  # Default volume

        # Filter by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        if len(data) == 0:
            raise ValueError("No data available for specified date range")

        return data.sort_index()

    def _reset_backtest_state(self):
        """Reset backtester to initial state."""
        self.current_capital = self.initial_capital
        self.current_position = 0.0
        self.current_position_price = 0.0
        self.trades.clear()
        self.state_transitions.clear()
        self.equity_curve.clear()
        self.state_counts.clear()
        self.state_durations.clear()
        self.transition_matrix.clear()

    def _record_state_transition(self,
                                timestamp: datetime,
                                from_state: int,
                                to_state: int,
                                price: float,
                                volume: float,
                                context: Dict[str, Any]):
        """Record a state transition."""
        transition = StateTransitionRecord(
            timestamp=timestamp,
            from_state=from_state,
            to_state=to_state,
            price=price,
            volume=volume,
            context_snapshot=context.copy()
        )
        self.state_transitions.append(transition)

        # Update transition matrix
        transition_key = (from_state, to_state)
        self.transition_matrix[transition_key] = self.transition_matrix.get(transition_key, 0) + 1

    def _execute_trade(self,
                      trade_signal: Dict[str, Any],
                      tick_data: pd.Series,
                      context: StateContext):
        """Execute a trade based on signal."""
        action = trade_signal.get('action', 'BUY').upper()
        entry_price = trade_signal.get('entry_price', tick_data['close'])

        # Calculate position size
        position_size = self._calculate_position_size(
            entry_price,
            trade_signal.get('stop_loss'),
            action
        )

        # Apply slippage
        slippage_amount = entry_price * (self.slippage_bps / 10000.0)
        if action == 'BUY':
            execution_price = entry_price + slippage_amount
        else:
            execution_price = entry_price - slippage_amount

        # Execute trade
        if action == 'BUY':
            self._open_long_position(execution_price, position_size, tick_data['timestamp'], trade_signal, context)
        else:
            self._open_short_position(execution_price, position_size, tick_data['timestamp'], trade_signal, context)

    def _calculate_position_size(self,
                                entry_price: float,
                                stop_loss: Optional[float],
                                action: str) -> float:
        """Calculate position size based on sizing method."""
        if self.position_sizing == 'fixed':
            return self.position_size

        elif self.position_sizing == 'percent_risk' and stop_loss:
            # Risk-based position sizing
            risk_per_unit = abs(entry_price - stop_loss)
            max_risk_amount = self.current_capital * self.max_position_risk

            if risk_per_unit > 0:
                return min(max_risk_amount / risk_per_unit, self.position_size * 100)
            else:
                return self.position_size

        else:
            return self.position_size

    def _open_long_position(self,
                           price: float,
                           size: float,
                           timestamp: datetime,
                           signal: Dict[str, Any],
                           context: StateContext):
        """Open a long position."""
        if self.current_position != 0:
            # Close existing position first
            self._close_position(price, timestamp, "position_change")

        self.current_position = size
        self.current_position_price = price

        # Deduct commission
        self.current_capital -= self.commission_per_trade

    def _open_short_position(self,
                            price: float,
                            size: float,
                            timestamp: datetime,
                            signal: Dict[str, Any],
                            context: StateContext):
        """Open a short position."""
        if self.current_position != 0:
            # Close existing position first
            self._close_position(price, timestamp, "position_change")

        self.current_position = -size
        self.current_position_price = price

        # Deduct commission
        self.current_capital -= self.commission_per_trade

    def _close_position(self,
                       price: float,
                       timestamp: datetime,
                       reason: str):
        """Close current position and record trade."""
        if self.current_position == 0:
            return

        # Calculate PnL
        if self.current_position > 0:  # Long position
            pnl = (price - self.current_position_price) * abs(self.current_position)
        else:  # Short position
            pnl = (self.current_position_price - price) * abs(self.current_position)

        # Apply commission and slippage
        total_commission = self.commission_per_trade * 2  # Entry + exit
        slippage_cost = price * (self.slippage_bps / 10000.0) * abs(self.current_position)
        net_pnl = pnl - total_commission - slippage_cost

        # Update capital
        self.current_capital += net_pnl

        # Record trade
        trade = TradeRecord(
            entry_time=getattr(self, '_position_entry_time', timestamp),
            exit_time=timestamp,
            entry_price=self.current_position_price,
            exit_price=price,
            action='BUY' if self.current_position > 0 else 'SELL',
            quantity=abs(self.current_position),
            pnl=net_pnl,
            pnl_pct=net_pnl / (self.current_position_price * abs(self.current_position)) * 100,
            commission=total_commission,
            slippage=slippage_cost,
            exit_reason=reason
        )
        self.trades.append(trade)

        # Reset position
        self.current_position = 0.0
        self.current_position_price = 0.0

    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value."""
        if self.current_position == 0:
            return self.current_capital

        # Mark-to-market unrealized PnL
        if self.current_position > 0:  # Long
            unrealized_pnl = (current_price - self.current_position_price) * self.current_position
        else:  # Short
            unrealized_pnl = (self.current_position_price - current_price) * abs(self.current_position)

        return self.current_capital + unrealized_pnl

    def _generate_results(self, start_date: datetime, end_date: datetime) -> BacktestResults:
        """Generate comprehensive backtest results."""
        # Convert equity curve to Series
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
            equity_series = pd.Series(equity_df['equity'].values, index=equity_df['timestamp'])
        else:
            equity_series = pd.Series([self.initial_capital], index=[start_date])

        # Calculate performance metrics
        total_return = (equity_series.iloc[-1] - self.initial_capital) / self.initial_capital

        # Calculate annualized return
        days = (end_date - start_date).days
        if days > 0:
            annual_return = (1 + total_return) ** (365.25 / days) - 1
        else:
            annual_return = 0.0

        # Calculate drawdown
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calculate Sharpe ratio (assuming daily data)
        returns = equity_series.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0

        # Trade statistics
        if self.trades:
            winning_trades = sum(1 for trade in self.trades if trade.pnl > 0)
            losing_trades = sum(1 for trade in self.trades if trade.pnl < 0)
            win_rate = winning_trades / len(self.trades) if self.trades else 0

            wins = [trade.pnl for trade in self.trades if trade.pnl > 0]
            losses = [trade.pnl for trade in self.trades if trade.pnl < 0]

            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            largest_win = max(wins) if wins else 0
            largest_loss = min(losses) if losses else 0

            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = largest_win = largest_loss = profit_factor = 0

        # State statistics
        state_stats = self._calculate_state_statistics()

        # Build transition matrix
        max_state = max(self.state_counts.keys()) if self.state_counts else 0
        transition_matrix = np.zeros((max_state + 1, max_state + 1))
        for (from_state, to_state), count in self.transition_matrix.items():
            if from_state <= max_state and to_state <= max_state:
                transition_matrix[from_state, to_state] = count

        return BacktestResults(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            state_statistics=state_stats,
            avg_states_per_trade=np.mean([t.states_traversed for t in self.trades]) if self.trades else 0,
            state_transition_matrix=transition_matrix,
            trades=self.trades.copy(),
            state_transitions=self.state_transitions.copy(),
            equity_curve=equity_series,
            drawdown_curve=drawdown
        )

    def _calculate_state_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Calculate detailed statistics for each state."""
        stats = {}

        for state in self.state_counts.keys():
            state_stats = {
                'count': self.state_counts[state],
                'percentage': self.state_counts[state] / sum(self.state_counts.values()) * 100
            }

            # Duration statistics
            if state in self.state_durations and self.state_durations[state]:
                durations = self.state_durations[state]
                state_stats.update({
                    'avg_duration_minutes': np.mean(durations),
                    'min_duration_minutes': np.min(durations),
                    'max_duration_minutes': np.max(durations),
                    'std_duration_minutes': np.std(durations)
                })

            stats[state] = state_stats

        return stats

    def plot_results(self, results: BacktestResults, save_path: Optional[str] = None) -> plt.Figure:
        """Generate comprehensive visualization of backtest results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MultiStateStrategy Backtest Results', fontsize=16)

        # Equity curve
        axes[0, 0].plot(results.equity_curve.index, results.equity_curve.values)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Portfolio Value')
        axes[0, 0].grid(True)

        # Drawdown
        axes[0, 1].fill_between(results.drawdown_curve.index,
                               results.drawdown_curve.values * 100, 0,
                               alpha=0.7, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True)

        # State transition heatmap
        if results.state_transition_matrix.sum() > 0:
            sns.heatmap(results.state_transition_matrix,
                       annot=True, fmt='d', cmap='Blues',
                       ax=axes[1, 0])
            axes[1, 0].set_title('State Transition Matrix')
            axes[1, 0].set_xlabel('To State')
            axes[1, 0].set_ylabel('From State')

        # Trade PnL distribution
        if results.trades:
            trade_pnls = [trade.pnl for trade in results.trades]
            axes[1, 1].hist(trade_pnls, bins=20, alpha=0.7)
            axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
            axes[1, 1].set_title('Trade PnL Distribution')
            axes[1, 1].set_xlabel('PnL')
            axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def export_results(self, results: BacktestResults, file_path: str):
        """Export backtest results to JSON file."""
        export_data = {
            'performance_metrics': {
                'total_return': results.total_return,
                'annual_return': results.annual_return,
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown': results.max_drawdown,
                'win_rate': results.win_rate,
                'profit_factor': results.profit_factor
            },
            'trade_statistics': {
                'total_trades': results.total_trades,
                'winning_trades': results.winning_trades,
                'losing_trades': results.losing_trades,
                'avg_win': results.avg_win,
                'avg_loss': results.avg_loss,
                'largest_win': results.largest_win,
                'largest_loss': results.largest_loss
            },
            'state_statistics': results.state_statistics,
            'avg_states_per_trade': results.avg_states_per_trade,
            'trades': [
                {
                    'entry_time': trade.entry_time.isoformat(),
                    'exit_time': trade.exit_time.isoformat(),
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'action': trade.action,
                    'quantity': trade.quantity,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'exit_reason': trade.exit_reason
                }
                for trade in results.trades
            ]
        }

        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)


def compare_strategies(strategies: List[Tuple[str, MultiStateStrategy]],
                      data: pd.DataFrame,
                      backtester_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Compare multiple MultiStateStrategy implementations.

    Args:
        strategies: List of (name, strategy) tuples
        data: Market data for backtesting
        backtester_config: Configuration for StateBacktester

    Returns:
        DataFrame comparing strategy performance metrics
    """
    if backtester_config is None:
        backtester_config = {}

    backtester = StateBacktester(**backtester_config)
    comparison_results = []

    for strategy_name, strategy in strategies:
        try:
            results = backtester.backtest_strategy(strategy, data)

            comparison_results.append({
                'Strategy': strategy_name,
                'Total Return (%)': results.total_return * 100,
                'Annual Return (%)': results.annual_return * 100,
                'Sharpe Ratio': results.sharpe_ratio,
                'Max Drawdown (%)': results.max_drawdown * 100,
                'Win Rate (%)': results.win_rate * 100,
                'Profit Factor': results.profit_factor,
                'Total Trades': results.total_trades,
                'Avg States/Trade': results.avg_states_per_trade
            })

        except Exception as e:
            print(f"Error backtesting {strategy_name}: {e}")
            continue

    return pd.DataFrame(comparison_results)