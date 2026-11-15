"""Risk calculations. Pure functions on NumPy arrays; no I/O.

Conventions
-----------
- `exposures`: currency value of each position (quantity x latest price); negative = short.
- `returns`: array of shape (n_days, n_assets) of simple daily returns.
- VaR and ES are reported as positive currency losses at confidence level `confidence`
  (e.g. 0.99) over `horizon_days`.
- P&L is linear in returns (no options), and positions are held fixed over the horizon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm


@dataclass(frozen=True)
class RiskResult:
    var: float
    expected_shortfall: float


def _validate(exposures: np.ndarray, returns: np.ndarray, confidence: float, horizon_days: int) -> None:
    if returns.ndim != 2 or returns.shape[1] != exposures.size:
        raise ValueError("returns must have one column per exposure")
    if returns.shape[0] < 2:
        raise ValueError("need at least 2 return observations")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")


def _tail(pnl: np.ndarray, confidence: float) -> RiskResult:
    """VaR and ES from a sample of P&L outcomes."""
    cutoff = np.quantile(pnl, 1.0 - confidence)
    tail = pnl[pnl <= cutoff]
    return RiskResult(var=float(-cutoff), expected_shortfall=float(-tail.mean()))


def historical(exposures, returns, confidence: float = 0.99, horizon_days: int = 1) -> RiskResult:
    """Historical simulation: replay every past day's returns on today's positions.

    Multi-day horizons scale the 1-day figures by sqrt(horizon), which assumes
    independent, identically distributed daily P&L.
    """
    exposures, returns = np.asarray(exposures, float), np.asarray(returns, float)
    _validate(exposures, returns, confidence, horizon_days)
    one_day = _tail(returns @ exposures, confidence)
    scale = math.sqrt(horizon_days)
    return RiskResult(one_day.var * scale, one_day.expected_shortfall * scale)


def parametric(exposures, returns, confidence: float = 0.99, horizon_days: int = 1) -> RiskResult:
    """Variance-covariance VaR: P&L assumed normal with the sample mean and covariance."""
    exposures, returns = np.asarray(exposures, float), np.asarray(returns, float)
    _validate(exposures, returns, confidence, horizon_days)
    mean, sigma = _moments(exposures, returns, horizon_days)
    z = norm.ppf(confidence)
    return RiskResult(
        var=float(-mean + z * sigma),
        expected_shortfall=float(-mean + sigma * norm.pdf(z) / (1.0 - confidence)),
    )


def monte_carlo(exposures, returns, confidence: float = 0.99, horizon_days: int = 1,
                n_sims: int = 100_000, rng: np.random.Generator | None = None) -> RiskResult:
    """Monte Carlo VaR: simulate horizon returns from a multivariate normal."""
    exposures, returns = np.asarray(exposures, float), np.asarray(returns, float)
    _validate(exposures, returns, confidence, horizon_days)
    rng = rng if rng is not None else np.random.default_rng()
    mu = returns.mean(axis=0) * horizon_days
    cov = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1)) * horizon_days
    sims = rng.multivariate_normal(mu, cov, size=n_sims, method="cholesky"
                                   if _is_positive_definite(cov) else "svd")
    return _tail(sims @ exposures, confidence)


def component_var(exposures, returns, confidence: float = 0.99, horizon_days: int = 1) -> np.ndarray:
    """Split parametric VaR across positions (Euler allocation).

    Component i = exposure_i x d(VaR)/d(exposure_i). The components add up to the
    portfolio's parametric VaR; a negative component is a position that hedges.
    """
    exposures, returns = np.asarray(exposures, float), np.asarray(returns, float)
    _validate(exposures, returns, confidence, horizon_days)
    cov = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1)) * horizon_days
    sigma = math.sqrt(float(exposures @ cov @ exposures))
    mean_part = -exposures * returns.mean(axis=0) * horizon_days
    if sigma == 0.0:
        return mean_part
    return mean_part + norm.ppf(confidence) * exposures * (cov @ exposures) / sigma


def _moments(exposures: np.ndarray, returns: np.ndarray, horizon_days: int) -> tuple[float, float]:
    cov = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1))
    mean = float(exposures @ returns.mean(axis=0)) * horizon_days
    sigma = math.sqrt(float(exposures @ cov @ exposures) * horizon_days)
    return mean, sigma


def _is_positive_definite(cov: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(cov)
        return True
    except np.linalg.LinAlgError:
        return False


# -- backtesting -----------------------------------------------------------------

@dataclass(frozen=True)
class BacktestResult:
    observations: int
    exceptions: int
    expected_exceptions: float
    kupiec_statistic: float
    p_value: float


def kupiec_test(exceptions: int, observations: int, confidence: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures test.

    H0: the true exception rate equals 1 - confidence. Returns the likelihood
    ratio statistic (chi-squared, 1 degree of freedom under H0) and its p-value.
    A small p-value means the VaR model has too many or too few exceptions.
    """
    if observations < 1 or not 0 <= exceptions <= observations:
        raise ValueError("need 0 <= exceptions <= observations and observations >= 1")
    p = 1.0 - confidence
    x, n = exceptions, observations

    def log_likelihood(rate: float) -> float:
        out = 0.0
        if n - x:
            out += (n - x) * math.log(1.0 - rate)
        if x:
            out += x * math.log(rate)
        return out

    stat = max(-2.0 * (log_likelihood(p) - log_likelihood(x / n)), 0.0)
    return stat, float(chi2.sf(stat, df=1))


def backtest_historical(exposures, returns, confidence: float = 0.99, window: int = 250) -> BacktestResult:
    """Rolling out-of-sample backtest of 1-day historical VaR.

    For each day after the first `window` days, VaR is estimated from the
    preceding `window` days only and compared with that day's realised P&L of
    the (fixed) current positions.
    """
    exposures, returns = np.asarray(exposures, float), np.asarray(returns, float)
    _validate(exposures, returns, confidence, 1)
    pnl = returns @ exposures
    if window < 20 or pnl.size <= window:
        raise ValueError("need window >= 20 and more observations than the window")
    exceptions = 0
    for t in range(window, pnl.size):
        var = -np.quantile(pnl[t - window:t], 1.0 - confidence)
        if pnl[t] < -var:
            exceptions += 1
    n = pnl.size - window
    stat, p_value = kupiec_test(exceptions, n, confidence)
    return BacktestResult(n, exceptions, n * (1.0 - confidence), stat, p_value)
