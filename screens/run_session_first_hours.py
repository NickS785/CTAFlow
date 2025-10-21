from __future__ import annotations

import argparse
from pathlib import Path

from screeners.session_first_hours import SessionFirstHoursParams, run_session_first_hours


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the session first-hours screener")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols to screen")
    parser.add_argument("--start", required=True, help="Session start date (inclusive)")
    parser.add_argument("--end", required=True, help="Session end date (inclusive)")
    parser.add_argument(
        "--lookback",
        type=int,
        default=20,
        help="Lookback days for volatility and time-of-day baselines",
    )
    parser.add_argument("--session-start", default="17:00", help="Session open time in HH:MM")
    parser.add_argument("--first-hours", type=int, default=2, help="Hours after the open to analyse")
    parser.add_argument("--bar-seconds", type=int, default=60, help="Aggregation interval in seconds")
    parser.add_argument("--tz", default="America/Chicago", help="Display timezone for sessions")
    parser.add_argument(
        "--scid-root",
        default=None,
        help="Directory with .scid files when SierraPy is unavailable",
    )
    parser.add_argument(
        "--min-bars",
        type=int,
        default=5,
        help="Minimum number of bars required in the analysis window",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output path (.csv or .parquet) for the screener result",
    )
    parser.add_argument("--tail", type=int, default=5, help="Number of most recent sessions to print")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    params = SessionFirstHoursParams(
        symbols=list(args.symbols),
        start_date=args.start,
        end_date=args.end,
        lookback_days=args.lookback,
        session_start_hhmm=args.session_start,
        first_hours=args.first_hours,
        bar_seconds=args.bar_seconds,
        tz_display=args.tz,
        scid_root=args.scid_root,
        min_bars_in_window=args.min_bars,
    )

    result = run_session_first_hours(params)
    if result.empty:
        print("No sessions matched the requested filters.")
        return

    print(result.tail(max(1, args.tail)))

    if args.out:
        out_path = Path(args.out)
        if out_path.suffix.lower() == ".parquet":
            result.to_parquet(out_path)
        else:
            result.to_csv(out_path)
        print(f"Saved screener output to {out_path.resolve()}")


if __name__ == "__main__":
    main()
