"""Load a demo portfolio into a running API.

The prices are SYNTHETIC: three correlated assets simulated from a multivariate
normal with fixed parameters. They are for trying the API, not market data.

    uvicorn risk_api.main:app --app-dir src &
    python3 scripts/seed_demo.py --url http://127.0.0.1:8000
"""

import argparse
import datetime as dt

import httpx
import numpy as np

SYMBOLS = ["AAA", "BBB", "CCC"]
START_PRICES = np.array([100.0, 50.0, 200.0])
DAILY_VOL = np.array([0.012, 0.020, 0.015])
CORRELATION = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.3], [0.2, 0.3, 1.0]])
QUANTITIES = {"AAA": 500, "BBB": 800, "CCC": -100}


def synthetic_prices(days: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cov = np.outer(DAILY_VOL, DAILY_VOL) * CORRELATION
    returns = rng.multivariate_normal(np.full(3, 0.0003), cov, size=days)
    return START_PRICES * np.cumprod(1.0 + returns, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--days", type=int, default=750)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    prices = synthetic_prices(args.days, args.seed)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(args.days)]
    with httpx.Client(base_url=args.url, timeout=30) as client:
        for col, symbol in enumerate(SYMBOLS):
            body = {"symbol": symbol,
                    "prices": [{"date": d.isoformat(), "close": float(p)} for d, p in zip(dates, prices[:, col])]}
            client.post("/prices", json=body).raise_for_status()
        created = client.post("/portfolios", json={"name": f"demo-{args.seed}"})
        created.raise_for_status()
        pid = created.json()["id"]
        client.put(f"/portfolios/{pid}/positions",
                   json=[{"symbol": s, "quantity": q} for s, q in QUANTITIES.items()]).raise_for_status()
    print(f"created portfolio {pid}; try GET {args.url}/portfolios/{pid}/risk")


if __name__ == "__main__":
    main()
