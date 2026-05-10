"""Dumitrescu-Hurlin (2012) panel Granger non-causality with cross-sectional
bootstrap (Lopez & Weber 2017, J. Stat. Software).

For each lag K, the country-i Wald test stat is W_i and the panel statistic is

    Wbar = (1/N) Σ W_i ;   Ztilde = sqrt(N/2K) * (Wbar - K) / sqrt(K)

Asymptotic p-values use a standard normal. For panels with cross-sectional
dependence (which our CD test confirms), the Lopez-Weber bootstrap re-samples
country residuals jointly under the no-causality null:

    1. Estimate restricted equation y_{i,t} = α_i + Σ_p γ_p y_{i,t-p} + e_{i,t}.
    2. Resample year-blocks of {e_{j,t} : j=1..N} with replacement, preserving
       cross-sectional structure within each year block.
    3. Build y* under the null with these e*; compute Wbar*, Ztilde*.
    4. Repeat B times; bootstrap p-value = share of replicates with stat
       at least as extreme as observed.

Test convention (DH 2012): right-tailed for both Wbar and Ztilde.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DHResult:
    direction: str
    lags: int
    Wbar: float
    Ztilde: float
    Ztilde_p_asym: float
    Ztilde_p_boot: float
    Wbar_p_boot: float
    n_countries: int
    n_obs: int
    B: int


def _country_wald(y: np.ndarray, x: np.ndarray, K: int) -> tuple[float, int]:
    """Country-i Wald test that all K x-lag coefficients are jointly zero
    in y_t = α + Σ γ_p y_{t-p} + Σ β_p x_{t-p} + e_t.

    Returns (Wald, valid_n)."""
    T = len(y)
    if T <= 2 * K + 2:
        return float("nan"), 0
    Y_lags = np.column_stack([y[K - p: T - p] for p in range(1, K + 1)])
    X_lags = np.column_stack([x[K - p: T - p] for p in range(1, K + 1)])
    Y = y[K:]
    n = len(Y)
    one = np.ones(n)
    Z = np.column_stack([one, Y_lags, X_lags])
    # Restricted: only own lags
    Z_r = np.column_stack([one, Y_lags])
    try:
        beta_u, *_ = np.linalg.lstsq(Z, Y, rcond=None)
        beta_r, *_ = np.linalg.lstsq(Z_r, Y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), 0
    res_u = Y - Z @ beta_u
    res_r = Y - Z_r @ beta_r
    rss_u = float(res_u @ res_u)
    rss_r = float(res_r @ res_r)
    df_resid = n - Z.shape[1]
    if rss_u <= 0 or df_resid <= 0:
        return float("nan"), 0
    F = ((rss_r - rss_u) / K) / (rss_u / df_resid)
    Wald = K * F  # Wald = K * F (chi^2_K-distributed under H0)
    return float(Wald), int(n)


def _panel_stat(panel_df: pd.DataFrame, y_col: str, x_col: str, K: int) -> tuple[float, float, int]:
    """Compute Wbar and Ztilde for a long DataFrame (iso3, year, y, x)."""
    Wis = []
    for _, g in panel_df.groupby("iso3"):
        if len(g) <= 2 * K + 2:
            continue
        W, _ = _country_wald(g[y_col].values, g[x_col].values, K)
        if not np.isnan(W):
            Wis.append(W)
    if not Wis:
        return float("nan"), float("nan"), 0
    N = len(Wis)
    Wbar = float(np.mean(Wis))
    Ztilde = np.sqrt(N / (2.0 * K)) * (Wbar - K) / 1.0
    return Wbar, Ztilde, N


def dh_test(
    panel: pd.DataFrame,
    y_col: str,
    x_col: str,
    K: int,
    direction_label: str,
    B: int = 500,
    seed: int = 42,
    difference: bool = True,
) -> DHResult:
    """Dumitrescu-Hurlin test with Lopez-Weber bootstrap.

    Per Hard Rule 1 of the brief: when both series are I(1) and not
    cointegrated (Phase 2 verdict), DH must be run on first differences,
    not levels. We therefore default `difference=True` and pre-difference
    the input series within each country. Set `difference=False` only if
    the caller has already differenced or if the variables are stationary
    in levels (e.g., `wgi_coc` post-2002, which is bounded -2.5..+2.5).

    Bootstrap procedure (year-block resampling, preserves cross-sectional
    correlation): we keep country IDs fixed and resample years with
    replacement.
    """
    rng = np.random.default_rng(seed)
    df = panel[["iso3", "year", y_col, x_col]].dropna().copy()
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    if difference:
        df[y_col] = df.groupby("iso3")[y_col].diff()
        df[x_col] = df.groupby("iso3")[x_col].diff()
        df = df.dropna(subset=[y_col, x_col]).reset_index(drop=True)

    Wbar_obs, Ztilde_obs, N = _panel_stat(df, y_col, x_col, K)
    if np.isnan(Wbar_obs):
        return DHResult(direction_label, K, float("nan"), float("nan"),
                        float("nan"), float("nan"), float("nan"), 0, len(df), 0)
    Ztilde_p_asym = float(1 - stats.norm.cdf(Ztilde_obs))

    if B <= 0:
        return DHResult(
            direction=direction_label, lags=K,
            Wbar=Wbar_obs, Ztilde=Ztilde_obs,
            Ztilde_p_asym=Ztilde_p_asym,
            Ztilde_p_boot=float("nan"),
            Wbar_p_boot=float("nan"),
            n_countries=N, n_obs=len(df), B=0,
        )

    # Build restricted residuals per country (own-lag-only model).
    countries = df.iso3.unique().tolist()
    resids = {}
    fits = {}
    y_lags_per = {}
    for iso, g in df.groupby("iso3"):
        if len(g) <= 2 * K + 2:
            continue
        T = len(g)
        Y_lags = np.column_stack([g[y_col].values[K - p: T - p] for p in range(1, K + 1)])
        Y = g[y_col].values[K:]
        Zr = np.column_stack([np.ones(len(Y)), Y_lags])
        try:
            br, *_ = np.linalg.lstsq(Zr, Y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        e = Y - Zr @ br
        resids[iso] = e
        fits[iso] = (br, g[y_col].values[:K].copy())
        y_lags_per[iso] = g[y_col].values

    # Year-block bootstrap: pivot residuals to a year × country matrix and
    # resample whole years with replacement.
    iso_idx = {iso: i for i, iso in enumerate(resids)}
    year_idx = sorted(df.year.unique())
    years_used = sorted({y for iso in resids for y in df[df.iso3 == iso].year.values[K:]})
    matr = np.full((len(years_used), len(resids)), np.nan)
    for iso, e in resids.items():
        years_iso = df[df.iso3 == iso].year.values[K:]
        for t, ev in zip(years_iso, e):
            if t in years_used:
                matr[years_used.index(t), iso_idx[iso]] = ev

    Wbar_b = []; Ztilde_b = []
    for b in range(B):
        chosen = rng.integers(0, len(years_used), size=len(years_used))
        # Build bootstrap dataset: for each country, replace residuals with
        # row-sampled residuals from `chosen` and reconstruct Y* recursively
        # using restricted parameters br.
        rows = []
        for iso, (br, init) in fits.items():
            j = iso_idx[iso]
            sampled_e = matr[chosen, j]
            mask = ~np.isnan(sampled_e)
            sampled_e = sampled_e[mask]
            if len(sampled_e) < K + 4:
                continue
            T_b = len(sampled_e) + K
            Y_star = np.empty(T_b); Y_star[:K] = init
            for t in range(K, T_b):
                pred = br[0] + sum(br[1 + p] * Y_star[t - 1 - p] for p in range(K))
                Y_star[t] = pred + sampled_e[t - K]
            X_iso = df[df.iso3 == iso][x_col].values[: T_b]
            if len(X_iso) != T_b:
                continue
            for t_, y_ in enumerate(Y_star):
                rows.append((iso, t_, y_, X_iso[t_]))
        if not rows:
            continue
        boot_df = pd.DataFrame(rows, columns=["iso3", "_t", y_col, x_col])
        boot_df["year"] = boot_df["_t"]
        Wb, Zb, _ = _panel_stat(boot_df.drop(columns="_t"), y_col, x_col, K)
        if not np.isnan(Wb):
            Wbar_b.append(Wb); Ztilde_b.append(Zb)

    if Wbar_b:
        Wbar_p_boot = float(np.mean(np.array(Wbar_b) >= Wbar_obs))
        Ztilde_p_boot = float(np.mean(np.array(Ztilde_b) >= Ztilde_obs))
    else:
        Wbar_p_boot = float("nan")
        Ztilde_p_boot = float("nan")

    return DHResult(
        direction=direction_label, lags=K,
        Wbar=Wbar_obs, Ztilde=Ztilde_obs,
        Ztilde_p_asym=Ztilde_p_asym,
        Ztilde_p_boot=Ztilde_p_boot,
        Wbar_p_boot=Wbar_p_boot,
        n_countries=N, n_obs=len(df), B=len(Wbar_b),
    )
