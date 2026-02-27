"""SB3 callbacks for RL trading metric tracking.

Provides ``TradingMetricsCallback`` which accumulates step-level PnL, drawdown,
exposure, and action statistics from V3ContinuousPPOEnv info dicts, then logs
a formatted summary every ``log_interval`` steps.

``EarlyStoppingCallback`` wraps TradingMetricsCallback and stops training when
PnL is still negative past a configurable checkpoint (default: halfway).
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TradingMetricsCallback(BaseCallback):
    """Logs rolling trading metrics during PPO training.

    Accumulates per-step info from the environment and prints a dashboard
    every ``log_interval`` global steps.  Tracks:

    * Cumulative PnL (gross and net of costs)
    * Max drawdown (over the window and lifetime)
    * Sharpe / Sortino (over the window)
    * Win rate, profit factor
    * Action distribution (Short / Flat / Long %)
    * Average exposure
    * Per-ticker breakdown (when multi-ticker)
    * Episode count and average episode reward

    Parameters
    ----------
    log_interval : int
        Print metrics every this many *global* timesteps (default 10_000).
    verbose : int
        0 = silent, 1 = summary line, 2 = full dashboard.
    """

    def __init__(self, log_interval: int = 10_000, verbose: int = 2):
        super().__init__(verbose=verbose)
        self.log_interval = log_interval
        self.stopped_early = False
        self._stop_reason: Optional[str] = None

        # Window accumulators (reset each log interval)
        self._w_rewards: List[float] = []
        self._w_gross_pnl: List[float] = []
        self._w_costs: List[float] = []
        self._w_actions: List[int] = []
        self._w_exposures: List[float] = []
        self._w_directions: List[bool] = []  # correct direction?
        self._w_ticker_pnl: Dict[str, List[float]] = defaultdict(list)

        # Lifetime accumulators (never reset)
        self._cum_pnl: float = 0.0
        self._cum_peak: float = 0.0
        self._max_dd_lifetime: float = 0.0
        self._all_net_pnl: List[float] = []  # every step's net PnL

        # Episode tracking
        self._ep_rewards: List[float] = []  # current episode
        self._ep_completed: int = 0
        self._ep_total_rewards: List[float] = []  # completed episodes in window

        self._last_log_step = 0
        self._start_time = None
        self._total_timesteps_planned: int = 0

    def _on_training_start(self) -> None:
        self._start_time = time.time()
        self._total_timesteps_planned = self.locals.get("total_timesteps", 0)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            r = float(rewards[i]) if i < len(rewards) else 0.0
            self._w_rewards.append(r)
            self._ep_rewards.append(r)

            gross = info.get("gross_pnl", 0.0)
            cost = info.get("cost", 0.0)
            self._w_gross_pnl.append(float(gross))
            self._w_costs.append(float(cost))

            # Action (0=short, 1=flat, 2=long)
            action = info.get("position_target", None)
            if action is not None:
                self._w_actions.append(int(action + 1))  # -1→0, 0→1, 1→2
            self._w_exposures.append(abs(info.get("position_target", 0.0)))

            # Direction accuracy
            prev_pos = info.get("position_prev", 0.0)
            step_ret = info.get("step_return", 0.0)
            if abs(prev_pos) > 1e-6:
                correct = (prev_pos > 0 and step_ret > 0) or (prev_pos < 0 and step_ret < 0)
                self._w_directions.append(correct)

            # Per-ticker
            ticker = info.get("ticker", "?")
            net_pnl = float(gross) - float(cost)
            self._w_ticker_pnl[ticker].append(net_pnl)
            self._all_net_pnl.append(net_pnl)

            # Lifetime cumulative PnL + drawdown
            self._cum_pnl += net_pnl
            self._cum_peak = max(self._cum_peak, self._cum_pnl)
            dd = self._cum_peak - self._cum_pnl
            self._max_dd_lifetime = max(self._max_dd_lifetime, dd)

            # Episode boundaries
            if i < len(dones) and dones[i]:
                self._ep_completed += 1
                self._ep_total_rewards.append(sum(self._ep_rewards))
                self._ep_rewards = []

        # Check if time to log
        if self.num_timesteps - self._last_log_step >= self.log_interval:
            self._log_metrics()
            self._reset_window()
            self._last_log_step = self.num_timesteps

        return True

    def _log_metrics(self) -> None:
        n = len(self._w_rewards)
        if n == 0:
            return

        rewards = np.array(self._w_rewards)
        gross = np.array(self._w_gross_pnl)
        costs = np.array(self._w_costs)
        net = gross - costs

        # Core stats
        reward_mean = float(np.mean(rewards))
        reward_std = float(np.std(rewards)) + 1e-8
        sharpe = reward_mean / reward_std

        downside = rewards[rewards < 0]
        down_std = float(np.sqrt(np.mean(np.square(downside)))) + 1e-8 if len(downside) else 1e-8
        sortino = reward_mean / down_std

        cum_net = np.cumsum(net)
        window_peak = np.maximum.accumulate(cum_net)
        window_dd = float(np.max(window_peak - cum_net)) if len(cum_net) else 0.0

        win_rate = float(np.mean(net > 0) * 100)
        profit = float(np.sum(net[net > 0]))
        loss = float(np.abs(np.sum(net[net < 0]))) + 1e-8
        pf = profit / loss

        avg_exposure = float(np.mean(self._w_exposures)) if self._w_exposures else 0.0
        dir_acc = float(np.mean(self._w_directions) * 100) if self._w_directions else 0.0

        # Action distribution
        actions = np.array(self._w_actions) if self._w_actions else np.array([])
        n_short = int(np.sum(actions == 0)) if len(actions) else 0
        n_flat = int(np.sum(actions == 1)) if len(actions) else 0
        n_long = int(np.sum(actions == 2)) if len(actions) else 0
        total_a = max(n_short + n_flat + n_long, 1)

        # Episode stats
        ep_avg = float(np.mean(self._ep_total_rewards)) if self._ep_total_rewards else 0.0
        ep_std = float(np.std(self._ep_total_rewards)) if len(self._ep_total_rewards) > 1 else 0.0

        # Timing
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        fps = self.num_timesteps / elapsed if elapsed > 0 else 0.0

        if self.verbose >= 2:
            print(f"\n{'=' * 70}")
            print(f"  Step {self.num_timesteps:>10,}  |  {n:,} steps in window  |  "
                  f"{fps:.0f} steps/s  |  {elapsed / 60:.1f}m elapsed")
            print(f"{'=' * 70}")
            print(f"  Reward     mean={reward_mean:+.4f}  std={reward_std - 1e-8:.4f}  "
                  f"sharpe={sharpe:+.3f}  sortino={sortino:+.3f}")
            print(f"  PnL        gross={float(np.sum(gross)):+.4f}  "
                  f"costs={float(np.sum(costs)):.4f}  "
                  f"net={float(np.sum(net)):+.4f}")
            print(f"  Window     maxDD={window_dd:.4f}  win%={win_rate:.1f}  PF={pf:.2f}  "
                  f"dir_acc={dir_acc:.1f}%")
            print(f"  Lifetime   cumPnL={self._cum_pnl:+.4f}  "
                  f"maxDD={self._max_dd_lifetime:.4f}")
            print(f"  Exposure   avg={avg_exposure:.3f}  |  "
                  f"Actions  S={n_short / total_a * 100:.0f}%  "
                  f"F={n_flat / total_a * 100:.0f}%  "
                  f"L={n_long / total_a * 100:.0f}%")
            print(f"  Episodes   completed={self._ep_completed}  "
                  f"avg_reward={ep_avg:+.2f}  std={ep_std:.2f}")

            # Per-ticker breakdown
            if len(self._w_ticker_pnl) > 1:
                print(f"  {'─' * 66}")
                for tkr in sorted(self._w_ticker_pnl.keys()):
                    t_pnl = np.array(self._w_ticker_pnl[tkr])
                    t_net = float(np.sum(t_pnl))
                    t_wr = float(np.mean(t_pnl > 0) * 100)
                    t_n = len(t_pnl)
                    print(f"  {tkr:6s}  net={t_net:+.4f}  win%={t_wr:.1f}  n={t_n}")
            print(f"{'─' * 70}")

        elif self.verbose == 1:
            print(f"[{self.num_timesteps:>8,}] "
                  f"reward={reward_mean:+.4f} sharpe={sharpe:+.3f} "
                  f"net={float(np.sum(net)):+.4f} maxDD={window_dd:.4f} "
                  f"exp={avg_exposure:.2f} "
                  f"S/F/L={n_short / total_a * 100:.0f}/{n_flat / total_a * 100:.0f}/{n_long / total_a * 100:.0f}")

        # Log to SB3's logger (shows up in TensorBoard if configured)
        if self.logger is not None:
            self.logger.record("trading/reward_mean", reward_mean)
            self.logger.record("trading/sharpe", sharpe)
            self.logger.record("trading/sortino", sortino)
            self.logger.record("trading/net_pnl_window", float(np.sum(net)))
            self.logger.record("trading/gross_pnl_window", float(np.sum(gross)))
            self.logger.record("trading/total_costs_window", float(np.sum(costs)))
            self.logger.record("trading/window_max_dd", window_dd)
            self.logger.record("trading/lifetime_cum_pnl", self._cum_pnl)
            self.logger.record("trading/lifetime_max_dd", self._max_dd_lifetime)
            self.logger.record("trading/win_rate", win_rate)
            self.logger.record("trading/profit_factor", pf)
            self.logger.record("trading/dir_accuracy", dir_acc)
            self.logger.record("trading/avg_exposure", avg_exposure)
            self.logger.record("trading/pct_short", n_short / total_a * 100)
            self.logger.record("trading/pct_flat", n_flat / total_a * 100)
            self.logger.record("trading/pct_long", n_long / total_a * 100)
            self.logger.record("trading/episodes_completed", self._ep_completed)
            self.logger.record("trading/ep_reward_mean", ep_avg)
            self.logger.record("trading/fps", fps)

    def _reset_window(self) -> None:
        self._w_rewards.clear()
        self._w_gross_pnl.clear()
        self._w_costs.clear()
        self._w_actions.clear()
        self._w_exposures.clear()
        self._w_directions.clear()
        self._w_ticker_pnl.clear()
        self._ep_total_rewards.clear()

    @property
    def progress(self) -> float:
        """Fraction of total planned timesteps completed (0.0 to 1.0)."""
        if self._total_timesteps_planned > 0:
            return self.num_timesteps / self._total_timesteps_planned
        return 0.0

    @property
    def recent_sharpe(self) -> float:
        """Sharpe ratio of the current (un-flushed) window rewards."""
        if len(self._w_rewards) < 2:
            return 0.0
        r = np.array(self._w_rewards)
        std = float(np.std(r)) + 1e-8
        return float(np.mean(r)) / std

    def get_lifetime_summary(self) -> Dict[str, float]:
        """Return lifetime stats (useful after training completes)."""
        return {
            "cum_pnl": self._cum_pnl,
            "max_drawdown": self._max_dd_lifetime,
            "episodes_completed": self._ep_completed,
            "total_timesteps": self.num_timesteps,
            "stopped_early": self.stopped_early,
            "stop_reason": self._stop_reason,
        }


class EarlyStoppingCallback(BaseCallback):
    """Stops training when trading metrics indicate a hopeless trial.

    Wraps a ``TradingMetricsCallback`` and evaluates stop conditions at
    configurable checkpoints.  When a condition triggers, it sets
    ``metrics_cb.stopped_early = True`` and returns ``False`` from
    ``_on_step`` which tells SB3 to halt ``model.learn()``.

    Default rules (all configurable):
      1. **Negative PnL at halfway** — if cumulative net PnL < 0 after 50%
         of total_timesteps, the config is unlikely to recover.
      2. **Flat exposure** — if avg exposure < ``min_exposure`` after the
         warmup fraction, the agent is doing nothing.
      3. **Drawdown blowup** — if lifetime max drawdown exceeds
         ``max_drawdown_threshold``, cut losses.

    Parameters
    ----------
    metrics_cb : TradingMetricsCallback
        The metrics callback to read stats from (must be in the same
        callback list passed to ``model.learn``).
    check_interval : int
        Evaluate stop conditions every this many steps (default 10_000).
    pnl_check_fraction : float
        Fraction of training at which cumPnL must be >= 0 (default 0.5).
    min_exposure : float
        Minimum avg exposure required after warmup (default 0.1).
    exposure_warmup_fraction : float
        Don't check exposure until this fraction of training (default 0.2).
    max_drawdown_threshold : float or None
        Absolute max drawdown that triggers a stop.  ``None`` = disabled.
    total_timesteps : int or None
        Legacy compatibility argument. If provided, overrides the SB3-derived
        total timestep count used for progress-based checks.
    max_drawdown : float or None
        Legacy alias for ``max_drawdown_threshold``.
    custom_rule : callable or None
        ``fn(metrics_cb, num_timesteps, total_timesteps) -> str | None``.
        Return a reason string to stop, or ``None`` to continue.
    verbose : int
        0 = silent, 1 = print stop reason.
    """

    def __init__(
        self,
        metrics_cb: TradingMetricsCallback,
        check_interval: int = 10_000,
        pnl_check_fraction: float = 0.5,
        min_exposure: float = 0.1,
        exposure_warmup_fraction: float = 0.2,
        max_drawdown_threshold: Optional[float] = None,
        total_timesteps: Optional[int] = None,
        max_drawdown: Optional[float] = None,
        custom_rule: Optional[Callable] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.metrics_cb = metrics_cb
        self.check_interval = check_interval
        self.pnl_check_fraction = pnl_check_fraction
        self.min_exposure = min_exposure
        self.exposure_warmup_fraction = exposure_warmup_fraction
        if max_drawdown_threshold is None:
            max_drawdown_threshold = max_drawdown
        self.max_drawdown_threshold = max_drawdown_threshold
        self.custom_rule = custom_rule
        self._last_check_step = 0
        self._total_timesteps: int = int(total_timesteps or 0)

    def _on_training_start(self) -> None:
        if self._total_timesteps <= 0:
            self._total_timesteps = self.locals.get("total_timesteps", 0)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_check_step < self.check_interval:
            return True
        self._last_check_step = self.num_timesteps

        if self._total_timesteps <= 0:
            return True

        progress = self.num_timesteps / self._total_timesteps
        mcb = self.metrics_cb

        # Rule 1: negative PnL at checkpoint
        if progress >= self.pnl_check_fraction and mcb._cum_pnl < 0:
            return self._stop(
                f"cumPnL={mcb._cum_pnl:+.4f} still negative at "
                f"{progress:.0%} progress ({self.num_timesteps:,} steps)"
            )

        # Rule 2: flat exposure (agent just sitting)
        if progress >= self.exposure_warmup_fraction and len(mcb._all_net_pnl) > 0:
            # Use recent window if available, else lifetime
            exposures = mcb._w_exposures if mcb._w_exposures else [0.0]
            avg_exp = float(np.mean(exposures))
            if avg_exp < self.min_exposure:
                return self._stop(
                    f"avg_exposure={avg_exp:.3f} < {self.min_exposure} at "
                    f"{progress:.0%} — agent is flat"
                )

        # Rule 3: drawdown blowup
        if self.max_drawdown_threshold is not None:
            if mcb._max_dd_lifetime > self.max_drawdown_threshold:
                return self._stop(
                    f"maxDD={mcb._max_dd_lifetime:.4f} > "
                    f"threshold={self.max_drawdown_threshold:.4f}"
                )

        # Rule 4: custom
        if self.custom_rule is not None:
            reason = self.custom_rule(mcb, self.num_timesteps, self._total_timesteps)
            if reason:
                return self._stop(reason)

        return True

    def _stop(self, reason: str) -> bool:
        self.metrics_cb.stopped_early = True
        self.metrics_cb._stop_reason = reason
        if self.verbose >= 1:
            print(f"  [EarlyStop] {reason}")
        return False
