"""SQLite persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolios (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS positions (
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol       TEXT    NOT NULL,
    quantity     REAL    NOT NULL,
    PRIMARY KEY (portfolio_id, symbol)
);
CREATE TABLE IF NOT EXISTS prices (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,          -- ISO 8601 date
    close  REAL NOT NULL CHECK (close > 0),
    PRIMARY KEY (symbol, date)
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Each request gets its own connection, but FastAPI may run the dependency and
    # the endpoint on different worker threads.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init(path: str | Path) -> None:
    conn = connect(path)
    conn.executescript(SCHEMA)
    conn.close()
