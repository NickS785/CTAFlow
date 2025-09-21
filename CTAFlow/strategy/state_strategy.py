#!/usr/bin/env python3
"""
Multi-state intraday trading strategy framework.

This module provides a base class for building trading systems that require multiple
stages of analysis before trade execution. Each stage represents a pre-condition
or context analysis step that must be satisfied to progress to the next stage.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging


class StateResult(Enum):
    """Possible outcomes of a state evaluation."""
    CONTINUE = "continue"      # Stay in current state
    ADVANCE = "advance"        # Move to next state
    RESET = "reset"           # Reset to initial state
    EXIT = "exit"             # Exit strategy (stop processing)


@dataclass
class StateContext:
    """Context information passed between states."""
    current_state: int
    entry_time: Optional[datetime] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_amount: Optional[float] = None
    position_size: Optional[float] = None

    # Custom context data for strategy-specific information
    custom_data: Dict[str, Any] = field(default_factory=dict)

    # State history for debugging and analysis
    state_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StateTransition:
    """Defines a state transition with conditions and actions."""
    from_state: int
    to_state: int
    condition: Callable[[pd.Series, StateContext], bool]
    action: Optional[Callable[[pd.Series, StateContext], None]] = None
    description: str = ""


class MultiStateStrategy(ABC):
    """
    Base class for multi-state trading strategies.

    This class provides a framework for building trading systems that analyze
    market context through multiple stages before executing trades. Each state
    represents a specific analysis phase with pre-conditions that must be met
    to progress to the next stage.

    Key Features:
    - State-based progression with configurable transitions
    - Context preservation across states
    - Risk management integration at each stage
    - Extensible framework for custom state logic
    - Built-in logging and state history tracking
    """

    def __init__(self,
                 symbol: str,
                 max_states: int = 5,
                 reset_timeout: Optional[timedelta] = None,
                 enable_logging: bool = True):
        """
        Initialize the multi-state strategy.

        Args:
            symbol: Trading symbol (e.g., 'CL_F', 'ES_F')
            max_states: Maximum number of states before auto-reset
            reset_timeout: Time limit before auto-reset (None = no timeout)
            enable_logging: Enable state transition logging
        """
        self.symbol = symbol
        self.max_states = max_states
        self.reset_timeout = reset_timeout
        self.enable_logging = enable_logging

        # Initialize state management
        self.context = StateContext(current_state=0)
        self.transitions: List[StateTransition] = []
        self.state_handlers: Dict[int, Callable] = {}

        # Setup logging
        if enable_logging:
            self.logger = logging.getLogger(f"{self.__class__.__name__}_{symbol}")
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = None

        # Initialize strategy-specific components
        self._setup_states()
        self._setup_transitions()

    @abstractmethod
    def _setup_states(self) -> None:
        """
        Setup state handlers for the strategy.

        Each state should be registered with self.register_state_handler().
        This method must be implemented by subclasses to define their
        specific state logic.
        """
        pass

    @abstractmethod
    def _setup_transitions(self) -> None:
        """
        Setup state transitions for the strategy.

        Define the conditions and actions for moving between states.
        This method must be implemented by subclasses.
        """
        pass

    def register_state_handler(self, state: int, handler: Callable) -> None:
        """Register a handler function for a specific state."""
        self.state_handlers[state] = handler

    def add_transition(self, transition: StateTransition) -> None:
        """Add a state transition rule."""
        self.transitions.append(transition)

    def process_tick(self, tick_data: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Process a single tick/bar of market data.

        Args:
            tick_data: Current market data (OHLCV + any additional fields)

        Returns:
            Optional trade signal dictionary or None
        """
        # Check for timeout reset
        if self._should_reset_on_timeout(tick_data):
            self._reset_strategy("Timeout reset")
            return None

        # Log current state
        if self.logger:
            self.logger.debug(f"Processing tick in state {self.context.current_state}")

        # Execute current state handler
        if self.context.current_state in self.state_handlers:
            state_result = self.state_handlers[self.context.current_state](tick_data, self.context)
        else:
            self.logger.warning(f"No handler for state {self.context.current_state}")
            state_result = StateResult.RESET

        # Process state result
        return self._process_state_result(state_result, tick_data)

    def _process_state_result(self, result: StateResult, tick_data: pd.Series) -> Optional[Dict[str, Any]]:
        """Process the result from a state handler."""
        timestamp = tick_data.get('timestamp', datetime.now())

        if result == StateResult.CONTINUE:
            # Stay in current state
            return None

        elif result == StateResult.ADVANCE:
            # Move to next state
            self._advance_state(tick_data)
            return None

        elif result == StateResult.RESET:
            # Reset to initial state
            self._reset_strategy("State-requested reset")
            return None

        elif result == StateResult.EXIT:
            # Exit strategy - generate trade signal
            trade_signal = self._generate_trade_signal(tick_data)
            self._reset_strategy("Trade executed")
            return trade_signal

        return None

    def _advance_state(self, tick_data: pd.Series) -> None:
        """Advance to the next state."""
        old_state = self.context.current_state
        self.context.current_state += 1

        # Record state transition
        self.context.state_history.append({
            'timestamp': tick_data.get('timestamp', datetime.now()),
            'from_state': old_state,
            'to_state': self.context.current_state,
            'price': tick_data.get('close', np.nan)
        })

        # Check for max states exceeded
        if self.context.current_state >= self.max_states:
            if self.logger:
                self.logger.warning(f"Max states ({self.max_states}) exceeded, resetting")
            self._reset_strategy("Max states exceeded")
            return

        if self.logger:
            self.logger.info(f"Advanced from state {old_state} to {self.context.current_state}")

    def _reset_strategy(self, reason: str) -> None:
        """Reset strategy to initial state."""
        if self.logger:
            self.logger.info(f"Resetting strategy: {reason}")

        # Preserve history if needed
        old_history = self.context.state_history.copy()

        # Reset context
        self.context = StateContext(current_state=0)

        # Optionally preserve some history for analysis
        if len(old_history) > 0:
            self.context.custom_data['last_run_history'] = old_history

    def _should_reset_on_timeout(self, tick_data: pd.Series) -> bool:
        """Check if strategy should reset due to timeout."""
        if self.reset_timeout is None or self.context.entry_time is None:
            return False

        current_time = tick_data.get('timestamp', datetime.now())
        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)

        time_elapsed = current_time - self.context.entry_time
        return time_elapsed > self.reset_timeout

    @abstractmethod
    def _generate_trade_signal(self, tick_data: pd.Series) -> Dict[str, Any]:
        """
        Generate trade signal when strategy reaches execution state.

        Args:
            tick_data: Current market data

        Returns:
            Dictionary containing trade signal information
        """
        pass

    # Utility methods for common state conditions
    def price_crossed_above(self, tick_data: pd.Series, level: float) -> bool:
        """Check if price crossed above a level."""
        current_price = tick_data.get('close', 0)
        previous_price = self.context.custom_data.get('previous_close', 0)
        return previous_price <= level < current_price

    def price_crossed_below(self, tick_data: pd.Series, level: float) -> bool:
        """Check if price crossed below a level."""
        current_price = tick_data.get('close', 0)
        previous_price = self.context.custom_data.get('previous_close', 0)
        return previous_price >= level > current_price

    def volume_spike(self, tick_data: pd.Series, threshold: float = 2.0) -> bool:
        """Check for volume spike above threshold."""
        current_volume = tick_data.get('volume', 0)
        avg_volume = self.context.custom_data.get('avg_volume', current_volume)
        return current_volume > (avg_volume * threshold)

    def time_in_range(self, tick_data: pd.Series, start_time: str, end_time: str) -> bool:
        """Check if current time is within specified range."""
        current_time = tick_data.get('timestamp', datetime.now())
        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)

        current_time_only = current_time.time()
        start = pd.to_datetime(start_time).time()
        end = pd.to_datetime(end_time).time()

        if start <= end:
            return start <= current_time_only <= end
        else:  # Overnight range
            return current_time_only >= start or current_time_only <= end

    def update_context(self, tick_data: pd.Series) -> None:
        """Update context with current tick data."""
        self.context.custom_data['previous_close'] = tick_data.get('close', 0)
        self.context.custom_data['previous_volume'] = tick_data.get('volume', 0)
        self.context.custom_data['last_update'] = tick_data.get('timestamp', datetime.now())

    # Analysis and debugging methods
    def get_current_state(self) -> int:
        """Get current state number."""
        return self.context.current_state

    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get history of state transitions."""
        return self.context.state_history.copy()

    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context."""
        return {
            'symbol': self.symbol,
            'current_state': self.context.current_state,
            'entry_time': self.context.entry_time,
            'entry_price': self.context.entry_price,
            'states_traversed': len(self.context.state_history),
            'custom_data_keys': list(self.context.custom_data.keys())
        }


class ExampleBreakoutStrategy(MultiStateStrategy):
    """
    Example implementation of a multi-state breakout strategy.

    States:
    0: Wait for consolidation period
    1: Confirm breakout with volume
    2: Wait for retest of breakout level
    3: Execute trade on successful retest
    """

    def __init__(self, symbol: str, **kwargs):
        self.breakout_level = None
        self.consolidation_periods = 0
        self.volume_threshold = 1.5
        super().__init__(symbol, max_states=4, **kwargs)

    def _setup_states(self) -> None:
        """Setup state handlers for breakout strategy."""
        self.register_state_handler(0, self._state_0_wait_consolidation)
        self.register_state_handler(1, self._state_1_confirm_breakout)
        self.register_state_handler(2, self._state_2_wait_retest)
        self.register_state_handler(3, self._state_3_execute_trade)

    def _setup_transitions(self) -> None:
        """Setup transitions - handled directly in state handlers for this example."""
        pass

    def _state_0_wait_consolidation(self, tick_data: pd.Series, context: StateContext) -> StateResult:
        """Wait for price consolidation period."""
        current_price = tick_data.get('close', 0)
        high = tick_data.get('high', current_price)
        low = tick_data.get('low', current_price)

        # Update price range tracking
        if 'price_high' not in context.custom_data:
            context.custom_data['price_high'] = high
            context.custom_data['price_low'] = low
            self.consolidation_periods = 0
        else:
            # Check if we're still in consolidation
            range_expansion = (high - low) / current_price
            if range_expansion < 0.005:  # Less than 0.5% range
                self.consolidation_periods += 1
                context.custom_data['price_high'] = max(context.custom_data['price_high'], high)
                context.custom_data['price_low'] = min(context.custom_data['price_low'], low)
            else:
                # Reset consolidation counter
                self.consolidation_periods = 0
                context.custom_data['price_high'] = high
                context.custom_data['price_low'] = low

        # Need at least 10 periods of consolidation
        if self.consolidation_periods >= 10:
            self.breakout_level = context.custom_data['price_high']
            return StateResult.ADVANCE

        self.update_context(tick_data)
        return StateResult.CONTINUE

    def _state_1_confirm_breakout(self, tick_data: pd.Series, context: StateContext) -> StateResult:
        """Confirm breakout with volume."""
        current_price = tick_data.get('close', 0)

        # Check for breakout above consolidation high
        if current_price > self.breakout_level:
            # Confirm with volume
            if self.volume_spike(tick_data, self.volume_threshold):
                context.entry_time = tick_data.get('timestamp', datetime.now())
                context.custom_data['breakout_price'] = current_price
                return StateResult.ADVANCE

        # Reset if price falls too far below breakout level
        if current_price < (self.breakout_level * 0.995):
            return StateResult.RESET

        self.update_context(tick_data)
        return StateResult.CONTINUE

    def _state_2_wait_retest(self, tick_data: pd.Series, context: StateContext) -> StateResult:
        """Wait for retest of breakout level."""
        current_price = tick_data.get('close', 0)

        # Check for retest (price coming back to breakout level)
        if abs(current_price - self.breakout_level) / self.breakout_level < 0.002:  # Within 0.2%
            return StateResult.ADVANCE

        # Reset if price moves too far away
        if current_price < (self.breakout_level * 0.99) or current_price > (self.breakout_level * 1.02):
            return StateResult.RESET

        self.update_context(tick_data)
        return StateResult.CONTINUE

    def _state_3_execute_trade(self, tick_data: pd.Series, context: StateContext) -> StateResult:
        """Execute trade after successful retest."""
        current_price = tick_data.get('close', 0)

        # Confirm price is moving away from retest level
        if current_price > (self.breakout_level * 1.001):  # 0.1% above breakout
            context.entry_price = current_price
            context.stop_loss = self.breakout_level * 0.995  # 0.5% below breakout
            context.take_profit = current_price * 1.01  # 1% profit target
            return StateResult.EXIT

        self.update_context(tick_data)
        return StateResult.CONTINUE

    def _generate_trade_signal(self, tick_data: pd.Series) -> Dict[str, Any]:
        """Generate trade signal for breakout strategy."""
        return {
            'symbol': self.symbol,
            'action': 'BUY',
            'entry_price': self.context.entry_price,
            'stop_loss': self.context.stop_loss,
            'take_profit': self.context.take_profit,
            'breakout_level': self.breakout_level,
            'signal_time': tick_data.get('timestamp', datetime.now()),
            'strategy': 'MultiStateBreakout',
            'states_traversed': len(self.context.state_history),
            'context': self.get_context_summary()
        }