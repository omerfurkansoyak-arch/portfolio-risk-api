# portfolio-risk-api

A REST service that stores portfolios and price history in SQLite and reports
Value-at-Risk (VaR) and Expected Shortfall (ES) three ways, with a per-position
risk breakdown and a statistical backtest of the VaR model.

Built with FastAPI, Pydantic v2, NumPy/SciPy and SQLite. The risk maths lives in
pure functions (`risk.py`) that are tested separately from the HTTP layer.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest

uvicorn risk_api.main:app --app-dir src          # interactive docs at http://127.0.0.1:8000/docs
python3 scripts/seed_demo.py                     # loads a demo portfolio with SYNTHETIC prices
```

The database path comes from the `RISK_DB` environment variable (default `data/risk.db`).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness check |
| POST | `/portfolios` | create a portfolio |
| GET | `/portfolios`, `/portfolios/{id}` | list / read |
| DELETE | `/portfolios/{id}` | delete a portfolio and its positions |
| PUT | `/portfolios/{id}/positions` | replace the position list (negative quantity = short) |
| POST | `/prices` | upload daily closes for a symbol (same date overwrites) |
| GET | `/portfolios/{id}/risk` | VaR, ES and per-position breakdown |
| GET | `/portfolios/{id}/backtest` | rolling VaR backtest with the Kupiec test |

`/risk` parameters: `method` (`historical`, `parametric`, `monte_carlo`), `confidence`
(0.5 to 1, default 0.99), `horizon_days` (1 to 250), `seed` (Monte Carlo only).

```bash
curl "http://127.0.0.1:8000/portfolios/1/risk?method=parametric&confidence=0.99"
```

## Methods

| Method | How | Assumption |
|---|---|---|
| Historical simulation | replay each past day's returns on today's positions, take the loss quantile | the past sample represents tomorrow; sqrt-of-time scaling for longer horizons |
| Parametric | `VaR = -mean + z * sqrt(e' Sigma e)` from the sample covariance | normal P&L |
| Monte Carlo | simulate horizon returns from a multivariate normal (Cholesky), take the loss quantile | normal returns |
| Component VaR | `e_i * dVaR/de_i` (Euler allocation); the parts sum to parametric VaR | normal P&L |
| Kupiec test | likelihood ratio test that the out-of-sample exception rate equals `1 - confidence` | independent exceptions |

## Example output

Demo portfolio from `scripts/seed_demo.py`: long 500 AAA, long 800 BBB, short 100 CCC,
749 daily returns of synthetic data, 99% confidence, 1 day.

| Method | VaR | Expected Shortfall |
|---|---|---|
| Historical | 5,321 | 5,881 |
| Parametric | 5,142 | 5,914 |
| Monte Carlo (seed 1) | 5,146 | 5,920 |

Component VaR: AAA 2,078, BBB 3,173, CCC -110. The short CCC position has a negative
component: it hedges the rest of the book.

Backtest (250-day window): 6 exceptions in 499 days against 4.99 expected, Kupiec
p-value 0.66, so the model is not rejected. That is expected here, because the demo
data is drawn from a fixed normal distribution; real returns are a harder test.

## Tests

25 tests. The risk functions are checked against closed forms (single-asset VaR, the
Kupiec statistic for 8 exceptions in 250 days), against each other (the three methods
agree within 3% on normal data), and for properties that must hold (diversification
and hedging reduce VaR, components sum to the total, sqrt-of-time scaling). The API
tests cover CRUD, validation errors, date alignment across symbols and reproducible
Monte Carlo seeds.

## Limitations

- Linear instruments only: P&L is quantity x price change, so options are not handled.
- Parametric and Monte Carlo VaR assume normal returns, which understates tail risk in real markets.
- One currency, no authentication, and SQLite is a single-writer store; this is a
  learning project, not a production risk system.
- Returns are computed between consecutive stored dates; gaps (weekends, holidays) are not annualised or adjusted.
