"""FastAPI application."""

# No `from __future__ import annotations` here: FastAPI resolves parameter annotations at
# runtime, and the `Conn` dependency alias is local to `create_app`.

import datetime as dt
import os
import sqlite3
from typing import Annotated, Iterator, Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query

from . import db, risk, schemas

MIN_OBSERVATIONS = 30


def create_app(db_path: str | None = None) -> FastAPI:
    path = db_path or os.environ.get("RISK_DB", "data/risk.db")
    db.init(path)
    app = FastAPI(title="Portfolio Risk API", version="0.1.0")

    def get_conn() -> Iterator[sqlite3.Connection]:
        conn = db.connect(path)
        try:
            yield conn
        finally:
            conn.close()

    Conn = Annotated[sqlite3.Connection, Depends(get_conn)]

    # -- helpers -------------------------------------------------------------

    def load_portfolio(conn: sqlite3.Connection, portfolio_id: int) -> schemas.Portfolio:
        row = conn.execute("SELECT id, name FROM portfolios WHERE id = ?", (portfolio_id,)).fetchone()
        if row is None:
            raise HTTPException(404, f"portfolio {portfolio_id} not found")
        positions = conn.execute(
            "SELECT symbol, quantity FROM positions WHERE portfolio_id = ? ORDER BY symbol",
            (portfolio_id,),
        ).fetchall()
        return schemas.Portfolio(
            id=row["id"], name=row["name"],
            positions=[schemas.Position(symbol=p["symbol"], quantity=p["quantity"]) for p in positions],
        )

    def market_data(conn: sqlite3.Connection, portfolio: schemas.Portfolio):
        """Return (symbols, exposures, returns, as_of) on the dates all symbols share."""
        if not portfolio.positions:
            raise HTTPException(409, "portfolio has no positions")
        symbols = [p.symbol for p in portfolio.positions]
        marks = ",".join("?" * len(symbols))
        rows = conn.execute(
            f"SELECT symbol, date, close FROM prices WHERE symbol IN ({marks})", symbols
        ).fetchall()
        missing = sorted(set(symbols) - {r["symbol"] for r in rows})
        if missing:
            raise HTTPException(409, f"no prices stored for: {', '.join(missing)}")
        frame = pd.DataFrame([tuple(r) for r in rows], columns=["symbol", "date", "close"])
        wide = frame.pivot(index="date", columns="symbol", values="close").sort_index().dropna()
        wide = wide[symbols]
        if len(wide) < MIN_OBSERVATIONS + 1:
            raise HTTPException(
                409, f"need at least {MIN_OBSERVATIONS + 1} dates common to all symbols, have {len(wide)}"
            )
        prices = wide.to_numpy(dtype=float)
        returns = prices[1:] / prices[:-1] - 1.0
        quantities = np.array([p.quantity for p in portfolio.positions])
        return symbols, quantities * prices[-1], returns, dt.date.fromisoformat(wide.index[-1])

    # -- routes --------------------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/portfolios", response_model=schemas.Portfolio, status_code=201)
    def create_portfolio(body: schemas.PortfolioCreate, conn: Conn):
        try:
            cur = conn.execute("INSERT INTO portfolios (name) VALUES (?)", (body.name,))
            conn.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(409, f"a portfolio named {body.name!r} already exists")
        return load_portfolio(conn, cur.lastrowid)

    @app.get("/portfolios", response_model=list[schemas.Portfolio])
    def list_portfolios(conn: Conn):
        ids = [r["id"] for r in conn.execute("SELECT id FROM portfolios ORDER BY id")]
        return [load_portfolio(conn, i) for i in ids]

    @app.get("/portfolios/{portfolio_id}", response_model=schemas.Portfolio)
    def get_portfolio(portfolio_id: int, conn: Conn):
        return load_portfolio(conn, portfolio_id)

    @app.delete("/portfolios/{portfolio_id}", status_code=204)
    def delete_portfolio(portfolio_id: int, conn: Conn):
        load_portfolio(conn, portfolio_id)
        conn.execute("DELETE FROM portfolios WHERE id = ?", (portfolio_id,))
        conn.commit()

    @app.put("/portfolios/{portfolio_id}/positions", response_model=schemas.Portfolio)
    def replace_positions(portfolio_id: int, body: list[schemas.Position], conn: Conn):
        """Replace the whole position list. Zero quantities are dropped."""
        load_portfolio(conn, portfolio_id)
        symbols = [p.symbol.upper() for p in body]
        if len(set(symbols)) != len(symbols):
            raise HTTPException(422, "each symbol may appear only once")
        with conn:
            conn.execute("DELETE FROM positions WHERE portfolio_id = ?", (portfolio_id,))
            conn.executemany(
                "INSERT INTO positions (portfolio_id, symbol, quantity) VALUES (?, ?, ?)",
                [(portfolio_id, s, p.quantity) for s, p in zip(symbols, body) if p.quantity != 0],
            )
        return load_portfolio(conn, portfolio_id)

    @app.post("/prices", response_model=schemas.PriceUploadResult)
    def upload_prices(body: schemas.PriceUpload, conn: Conn):
        """Store daily closes. Re-uploading a date overwrites it."""
        symbol = body.symbol.upper()
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO prices (symbol, date, close) VALUES (?, ?, ?)",
                [(symbol, p.date.isoformat(), p.close) for p in body.prices],
            )
        return schemas.PriceUploadResult(symbol=symbol, stored=len(body.prices))

    @app.get("/portfolios/{portfolio_id}/risk", response_model=schemas.RiskReport)
    def portfolio_risk(
        portfolio_id: int,
        conn: Conn,
        method: Literal["historical", "parametric", "monte_carlo"] = "historical",
        confidence: Annotated[float, Query(gt=0.5, lt=1.0)] = 0.99,
        horizon_days: Annotated[int, Query(ge=1, le=250)] = 1,
        seed: Annotated[int | None, Query(description="Monte Carlo seed, for reproducible results")] = None,
    ):
        portfolio = load_portfolio(conn, portfolio_id)
        symbols, exposures, returns, as_of = market_data(conn, portfolio)
        if method == "historical":
            result = risk.historical(exposures, returns, confidence, horizon_days)
        elif method == "parametric":
            result = risk.parametric(exposures, returns, confidence, horizon_days)
        else:
            result = risk.monte_carlo(exposures, returns, confidence, horizon_days,
                                      rng=np.random.default_rng(seed))
        components = risk.component_var(exposures, returns, confidence, horizon_days)
        return schemas.RiskReport(
            portfolio_id=portfolio_id, method=method, confidence=confidence,
            horizon_days=horizon_days, observations=len(returns), as_of=as_of,
            market_value=float(exposures.sum()), var=result.var,
            expected_shortfall=result.expected_shortfall,
            positions=[
                schemas.PositionRisk(symbol=s, exposure=float(e), component_var=float(c))
                for s, e, c in zip(symbols, exposures, components)
            ],
        )

    @app.get("/portfolios/{portfolio_id}/backtest", response_model=schemas.BacktestReport)
    def portfolio_backtest(
        portfolio_id: int,
        conn: Conn,
        confidence: Annotated[float, Query(gt=0.5, lt=1.0)] = 0.99,
        window: Annotated[int, Query(ge=20, le=2000)] = 250,
    ):
        portfolio = load_portfolio(conn, portfolio_id)
        _, exposures, returns, _ = market_data(conn, portfolio)
        if len(returns) <= window:
            raise HTTPException(409, f"backtest needs more than {window} return observations, have {len(returns)}")
        result = risk.backtest_historical(exposures, returns, confidence, window)
        return schemas.BacktestReport(
            portfolio_id=portfolio_id, confidence=confidence, window=window,
            observations=result.observations, exceptions=result.exceptions,
            expected_exceptions=result.expected_exceptions,
            kupiec_statistic=result.kupiec_statistic, p_value=result.p_value,
            model_rejected_at_5pct=result.p_value < 0.05,
        )

    return app
