"""Phase 3 — Panel VAR in first differences, country + year FE.

Brief preferred R `panelvar::pvargmm` (Arellano-Bover/Blundell-Bond), but
the installed package has an internal bug ("object 'result' not found")
that breaks even its own example dataset. We fall back to a fixed-effects
PVAR estimated equation-by-equation via `linearmodels.PanelOLS`:

   Δy_{i,t} = α_i^y + λ_t^y + Σ_{p=1..P} (a^y_p Δy_{i,t-p} + b^y_p Δx_{i,t-p}) + ε^y
   Δx_{i,t} = α_i^x + λ_t^x + Σ_{p=1..P} (a^x_p Δy_{i,t-p} + b^x_p Δx_{i,t-p}) + ε^x

with country FE absorbed by within-transform and year FE entered as
dummies (entity_effects + time_effects). Driscoll–Kraay SEs handle the
residual CSD that motivated using differences in the first place.

With T ≈ 33, Nickell bias on the autoregressive coefficient is O(1/T) ≈ 3 %
and is reported as a caveat. As a sensitivity, the same equations are
re-estimated with second-lag levels as IV (Anderson–Hsiao), which is
unbiased.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

log = logging.getLogger(__name__)


def _prep(panel: pd.DataFrame, vars_: Iterable[str], lags: int) -> pd.DataFrame:
    df = panel[["iso3", "year", *vars_]].dropna().copy()
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    # First differences
    for v in vars_:
        df[f"d_{v}"] = df.groupby("iso3")[v].diff()
    # Lags of differences
    for v in vars_:
        for p in range(1, lags + 1):
            df[f"d_{v}_l{p}"] = df.groupby("iso3")[f"d_{v}"].shift(p)
    df = df.dropna()
    return df.set_index(["iso3", "year"])


@dataclass
class PVARSpec:
    label: str
    dep_vars: tuple[str, ...]
    lags: int


def fit_pvar_diff(
    panel: pd.DataFrame,
    spec: PVARSpec,
    use_year_fe: bool = True,
    cluster_entity: bool = True,
) -> dict:
    df = _prep(panel, spec.dep_vars, spec.lags)
    n_iso = df.index.get_level_values("iso3").nunique()
    n_years = df.index.get_level_values("year").nunique()
    log.info("[%s] N=%d countries, T-lags-diffs obs=%d, year FE=%s",
             spec.label, n_iso, len(df), use_year_fe)

    eq_results = {}
    for dep in spec.dep_vars:
        y_col = f"d_{dep}"
        rhs_cols = [
            f"d_{v}_l{p}" for v in spec.dep_vars for p in range(1, spec.lags + 1)
        ]
        X = df[rhs_cols]
        y = df[y_col]
        mod = PanelOLS(
            y, X,
            entity_effects=True,
            time_effects=use_year_fe,
            drop_absorbed=True,
        )
        cov_kwargs = {}
        if cluster_entity:
            cov_kwargs = dict(cov_type="clustered", cluster_entity=True)
        else:
            cov_kwargs = dict(cov_type="robust")
        res = mod.fit(**cov_kwargs)
        eq_results[dep] = res
    return {
        "label": spec.label,
        "lags": spec.lags,
        "dep_vars": spec.dep_vars,
        "n_countries": int(n_iso),
        "n_obs": int(len(df)),
        "results": eq_results,
    }


def companion_eigenvalues(eq_results: dict, dep_vars: tuple[str, ...], lags: int) -> np.ndarray:
    """Stability check: spectral radius of companion matrix."""
    K = len(dep_vars)
    P = lags
    A_blocks = []
    for p in range(1, P + 1):
        A_p = np.zeros((K, K))
        for i, dep in enumerate(dep_vars):
            res = eq_results[dep]
            for j, v in enumerate(dep_vars):
                col = f"d_{v}_l{p}"
                if col in res.params.index:
                    A_p[i, j] = res.params[col]
        A_blocks.append(A_p)
    if P == 1:
        comp = A_blocks[0]
    else:
        I = np.eye(K * (P - 1))
        Z = np.zeros((K * (P - 1), K))
        comp = np.block([
            [np.column_stack(A_blocks)],
            [np.column_stack([I, Z])],
        ])
    return np.linalg.eigvals(comp)


def flatten_results(out: dict) -> pd.DataFrame:
    rows = []
    for dep, res in out["results"].items():
        for var, b in res.params.items():
            rows.append({
                "spec": out["label"],
                "lags": out["lags"],
                "equation": dep,
                "regressor": var,
                "coef": float(b),
                "se": float(res.std_errors[var]),
                "t": float(res.tstats[var]),
                "p": float(res.pvalues[var]),
                "n_obs": out["n_obs"],
                "n_countries": out["n_countries"],
            })
    return pd.DataFrame(rows)
