import datetime as dt

import numpy as np
import pytest
from fastapi.testclient import TestClient

from risk_api.app import create_app


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(str(tmp_path / "test.db")))


def upload(client, symbol, closes, start=dt.date(2025, 1, 1)):
    body = {"symbol": symbol,
            "prices": [{"date": (start + dt.timedelta(days=i)).isoformat(), "close": float(c)}
                       for i, c in enumerate(closes)]}
    return client.post("/prices", json=body)


def series(seed, n=400, vol=0.01, start=100.0):
    rng = np.random.default_rng(seed)
    return start * np.cumprod(1 + rng.normal(0, vol, n))


def make_portfolio(client, positions, name="p"):
    pid = client.post("/portfolios", json={"name": name}).json()["id"]
    assert client.put(f"/portfolios/{pid}/positions", json=positions).status_code == 200
    return pid


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_portfolio_crud(client):
    created = client.post("/portfolios", json={"name": "main"})
    assert created.status_code == 201
    pid = created.json()["id"]
    assert client.post("/portfolios", json={"name": "main"}).status_code == 409
    body = client.put(f"/portfolios/{pid}/positions",
                      json=[{"symbol": "aaa", "quantity": 10}, {"symbol": "BBB", "quantity": 0}]).json()
    assert body["positions"] == [{"symbol": "AAA", "quantity": 10.0}]   # upper-cased, zero dropped
    assert len(client.get("/portfolios").json()) == 1
    assert client.delete(f"/portfolios/{pid}").status_code == 204
    assert client.get(f"/portfolios/{pid}").status_code == 404


def test_duplicate_symbols_rejected(client):
    pid = client.post("/portfolios", json={"name": "d"}).json()["id"]
    r = client.put(f"/portfolios/{pid}/positions",
                   json=[{"symbol": "AAA", "quantity": 1}, {"symbol": "aaa", "quantity": 2}])
    assert r.status_code == 422


def test_price_upload_overwrites_same_date(client):
    upload(client, "AAA", [100, 101])
    upload(client, "AAA", [100, 105])
    upload(client, "AAA", series(1))
    pid = make_portfolio(client, [{"symbol": "AAA", "quantity": 1}])
    assert client.get(f"/portfolios/{pid}/risk").json()["observations"] == 399


def test_invalid_price_rejected(client):
    assert upload(client, "AAA", [100, -5]).status_code == 422


@pytest.mark.parametrize("method", ["historical", "parametric", "monte_carlo"])
def test_risk_report(client, method):
    upload(client, "AAA", series(1))
    upload(client, "BBB", series(2, vol=0.02, start=50))
    pid = make_portfolio(client, [{"symbol": "AAA", "quantity": 100}, {"symbol": "BBB", "quantity": -40}])
    r = client.get(f"/portfolios/{pid}/risk", params={"method": method, "confidence": 0.95, "seed": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == method and body["observations"] == 399
    assert 0 < body["var"] < body["expected_shortfall"]
    exposures = {p["symbol"]: p["exposure"] for p in body["positions"]}
    assert exposures["AAA"] > 0 > exposures["BBB"]
    assert body["market_value"] == pytest.approx(sum(exposures.values()))


def test_component_var_sums_to_parametric_var(client):
    upload(client, "AAA", series(1))
    upload(client, "BBB", series(2))
    pid = make_portfolio(client, [{"symbol": "AAA", "quantity": 100}, {"symbol": "BBB", "quantity": 50}])
    body = client.get(f"/portfolios/{pid}/risk", params={"method": "parametric"}).json()
    assert sum(p["component_var"] for p in body["positions"]) == pytest.approx(body["var"])


def test_monte_carlo_seed_is_reproducible(client):
    upload(client, "AAA", series(1))
    pid = make_portfolio(client, [{"symbol": "AAA", "quantity": 100}])
    get = lambda: client.get(f"/portfolios/{pid}/risk", params={"method": "monte_carlo", "seed": 9}).json()["var"]
    assert get() == get()


def test_only_common_dates_are_used(client):
    upload(client, "AAA", series(1, n=400))
    upload(client, "BBB", series(2, n=100), start=dt.date(2025, 6, 1))   # overlaps the last part of AAA
    pid = make_portfolio(client, [{"symbol": "AAA", "quantity": 1}, {"symbol": "BBB", "quantity": 1}])
    assert client.get(f"/portfolios/{pid}/risk").json()["observations"] == 99


def test_risk_errors(client):
    assert client.get("/portfolios/999/risk").status_code == 404
    empty = client.post("/portfolios", json={"name": "empty"}).json()["id"]
    assert client.get(f"/portfolios/{empty}/risk").status_code == 409
    pid = make_portfolio(client, [{"symbol": "ZZZ", "quantity": 1}], name="noprices")
    r = client.get(f"/portfolios/{pid}/risk")
    assert r.status_code == 409 and "ZZZ" in r.json()["detail"]
    upload(client, "ZZZ", series(3, n=10))
    assert client.get(f"/portfolios/{pid}/risk").status_code == 409          # too little history
    upload(client, "ZZZ", series(3))
    assert client.get(f"/portfolios/{pid}/risk", params={"confidence": 1.2}).status_code == 422
    assert client.get(f"/portfolios/{pid}/risk", params={"method": "magic"}).status_code == 422


def test_backtest_endpoint(client):
    upload(client, "AAA", series(4, n=900))
    pid = make_portfolio(client, [{"symbol": "AAA", "quantity": 10}])
    body = client.get(f"/portfolios/{pid}/backtest", params={"window": 250}).json()
    assert body["observations"] == 899 - 250
    assert body["expected_exceptions"] == pytest.approx(6.49)
    assert 0 <= body["p_value"] <= 1
    short = client.get(f"/portfolios/{pid}/backtest", params={"window": 1000})
    assert short.status_code == 409
