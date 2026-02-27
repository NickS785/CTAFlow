"""SB3 callbacks for RL trading metric tracking.

Provides ``TradingMetricsCallback`` which accumulates step-level PnL, drawdown,
exposure, and action statistics from V3ContinuousPPOEnv info dicts, then logs
a formatted summary every ``log_interval`` steps.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional

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

        # Episode tracking
        self._ep_rewards: List[float] = []  # current episode
        self._ep_completed: int = 0
        self._ep_total_rewards: List[float] = []  # completed episodes in window

        self._last_log_step = 0
        self._start_time = None

    def _on_training_start(self) -> None:
        self._start_time = time.time()

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

    def get_lifetime_summary(self) -> Dict[str, float]:
        """Return lifetime stats (useful after training completes)."""
        return {
            "cum_pnl": self._cum_pnl,
            "max_drawdown": self._max_dd_lifetime,
            "episodes_completed": self._ep_completed,
            "total_timesteps": self.num_timesteps,
        }
