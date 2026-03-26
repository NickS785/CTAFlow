#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import ssl
import sys
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, build_opener, HTTPSHandler


def _build_opener(verify_ssl: bool):
    if verify_ssl:
        return build_opener()

    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return build_opener(HTTPSHandler(context=context))


def _request_json(
    opener,
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Any:
    if params:
        url = f"{url}?{urlencode(params)}"

    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url=url, data=data, headers=headers, method=method.upper())
    with opener.open(req, timeout=15) as resp:
        body = resp.read().decode("utf-8")
        if not body:
            return {}
        return json.loads(body)


def _print_json(label: str, data: Any) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal IBKR Client Portal API smoke test.")
    parser.add_argument("--gateway-url", default="https://localhost:5000")
    parser.add_argument("--conid", default="269460054")
    parser.add_argument("--account", default="")
    parser.add_argument("--period", default="2d")
    parser.add_argument("--bar", default="1min")
    parser.add_argument("--verify-ssl", action="store_true")
    args = parser.parse_args()

    base = args.gateway_url.rstrip("/") + "/v1/api"
    opener = _build_opener(args.verify_ssl)

    try:
        tickle = _request_json(opener, "POST", f"{base}/tickle", payload={})
        _print_json("tickle", tickle)

        history = _request_json(
            opener,
            "GET",
            f"{base}/iserver/marketdata/history",
            params={
                "conid": args.conid,
                "period": args.period,
                "bar": args.bar,
                "outsideRth": "true",
            },
        )
        bars = history.get("data", []) if isinstance(history, dict) else []
        print(f"\nHistory bars returned: {len(bars)}")
        if bars:
            print("First bar:")
            print(json.dumps(bars[0], indent=2, sort_keys=True, default=str))
            print("Last bar:")
            print(json.dumps(bars[-1], indent=2, sort_keys=True, default=str))
        else:
            _print_json("history", history)

        snapshot = _request_json(
            opener,
            "GET",
            f"{base}/iserver/marketdata/snapshot",
            params={
                "conids": args.conid,
                "fields": "31,84,86,7295",
            },
        )
        _print_json("snapshot", snapshot)

        if args.account:
            positions = _request_json(
                opener,
                "GET",
                f"{base}/portfolio/{args.account}/positions/0",
            )
            _print_json("positions", positions)

        print("\nIBKR minimal smoke test completed.")
        return 0
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTPError {exc.code}: {exc.reason}", file=sys.stderr)
        if body:
            print(body, file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"URLError: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
