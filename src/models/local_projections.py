"""Phase 5 — Panel local projections (Jordà 2005).

For h = 0..H, estimate
   Δy_{i,t+h} = β_h × shock_{i,t} + γ_h' Z_{i,t} + α_i + λ_t + ε_{i,t+h}

with country and year FE and Driscoll–Kraay SEs (cluster_entity for now).

Two definitions of `shock`:
  - residual from country-specific AR(1) of the shock variable, OR
  - one-standard-deviation shock = year-on-year change > 1*sd of country.
We use the AR(1) residual ("orthogonalised shock"), which is more standard
and lets us interpret β_h as the impulse response to a unit innovation.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

log = logging.getLogger(__name__)


def country_ar1_residual(panel: pd.DataFrame, var: str) -> pd.Series:
    """Per-country AR(1) residual of `var`."""
    out = pd.Series(index=panel.index, dtype=float)
    for iso, g in panel.groupby("iso3"):
        s = g[var]
        idx = s.index
        s_lag = s.shift(1)
        valid = s_lag.notna() & s.notna()
        if valid.sum() < 4:
            continue
        x = s_lag[valid].values
        y = s[valid].values
        ones = np.ones_like(x)
        X = np.column_stack([ones, x])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ b
        out.loc[idx[valid]] = resid
    return out


def lp_irf(
    panel: pd.DataFrame,
    shock_var: str,
    response_var: str,
    horizons: Iterable[int] = range(0, 6),
    controls: Iterable[str] = (),
) -> pd.DataFrame:
    """Estimate β_h for h in `horizons`.

    Response is the cumulative change Δ^h y_{i,t+h} ≡ y_{i,t+h} - y_{i,t-1}.
    Returns a DataFrame with horizon, beta, se, t, p, n_obs, n_countries, ci_low, ci_high.
    """
    df = panel[["iso3", "year", shock_var, response_var, *controls]].copy()
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    df["shock"] = country_ar1_residual(df, shock_var)
    df = df.set_index(["iso3", "year"])

    rows = []
    for h in horizons:
        df_h = df.copy()
        df_h["resp"] = df_h.groupby(level="iso3")[response_var].shift(-h)
        df_h["resp_lag1"] = df_h.groupby(level="iso3")[response_var].shift(1)
        df_h["dep"] = df_h["resp"] - df_h["resp_lag1"]
        df_h = df_h.dropna(subset=["dep", "shock"])
        rhs = ["shock", *controls, "resp_lag1"]
        rhs = [c for c in rhs if c in df_h.columns]
        df_h = df_h.dropna(subset=rhs)
        X = df_h[rhs]
        y = df_h["dep"]
        try:
            mod = PanelOLS(y, X, entity_effects=True, time_effects=True,
                           drop_absorbed=True)
            res = mod.fit(cov_type="clustered", cluster_entity=True)
        except Exception as e:
            log.warning("h=%d failed: %s", h, e)
            continue
        beta = float(res.params.get("shock", np.nan))
        se = float(res.std_errors.get("shock", np.nan))
        t = float(res.tstats.get("shock", np.nan))
        p = float(res.pvalues.get("shock", np.nan))
        rows.append({
            "horizon": h,
            "beta": beta, "se": se, "t": t, "p": p,
            "ci_low": beta - 1.96 * se,
            "ci_high": beta + 1.96 * se,
            "n_obs": int(res.nobs),
            "n_countries": int(df_h.index.get_level_values("iso3").nunique()),
        })
    return pd.DataFrame(rows)
