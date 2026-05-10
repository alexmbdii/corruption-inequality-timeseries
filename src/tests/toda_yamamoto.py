"""Toda-Yamamoto (1995) country-by-country Granger non-causality test.

Procedure (per Toda & Yamamoto 1995):
  1. Determine d_max = max integration order of (y, x) by ADF/KPSS.
  2. Choose VAR lag k on levels by AIC (max 4 with this T).
  3. Estimate VAR(k + d_max) on levels.
  4. Wald-test that the FIRST k coefficients of x in the y equation
     (and vice versa) are jointly zero, using only the first k lags' SEs
     (the extra d_max lags purely soak up potential unit-root nuisance).
  5. The Wald statistic is asymptotically χ²_k under the null.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class TYResult:
    iso3: str
    direction: str  # 'corr_to_gini' or 'gini_to_corr'
    k: int
    d_max: int
    wald: float
    p_value: float
    n_obs: int


def _integration_order(s: pd.Series) -> int:
    s = pd.Series(s).dropna().reset_index(drop=True)
    if len(s) < 12:
        return 0
    try:
        adf_p = adfuller(s, autolag="AIC")[1]
        kpss_p = kpss(s, regression="c", nlags="auto")[1]
    except Exception:
        return 1
    # If ADF rejects (stationary) AND KPSS does not reject (stationary): I(0)
    if adf_p < 0.05 and kpss_p > 0.05:
        return 0
    return 1


def _var_aic_select(Y: np.ndarray, k_max: int = 4) -> int:
    n_obs, K = Y.shape
    best_k = 1
    best_aic = np.inf
    for k in range(1, k_max + 1):
        if n_obs - k < K * k + 1:
            continue
        Yt = Y[k:]
        X = np.column_stack(
            [Y[k - p: -p if p > 0 else None] for p in range(1, k + 1)]
        )
        X = np.column_stack([np.ones(len(Yt)), X])
        try:
            B, *_ = np.linalg.lstsq(X, Yt, rcond=None)
            res = Yt - X @ B
            sigma2 = (res.T @ res) / (n_obs - k)
            aic = np.log(np.abs(np.linalg.det(sigma2))) + 2 * (X.shape[1] * K) / (n_obs - k)
            if aic < best_aic:
                best_aic = aic; best_k = k
        except np.linalg.LinAlgError:
            continue
    return best_k


def ty_country(
    df_country: pd.DataFrame, y_col: str, x_col: str, k_max: int = 4
) -> tuple[float, float, int, int, int]:
    """Run TY for a single country. Returns (Wald, p, k, d_max, n_obs)."""
    df = df_country[[y_col, x_col]].dropna().reset_index(drop=True)
    n = len(df)
    if n < 18:
        return (float("nan"), float("nan"), 0, 0, n)
    d_max = max(_integration_order(df[y_col]), _integration_order(df[x_col]))
    Y = df[[y_col, x_col]].values
    k = _var_aic_select(Y, k_max=k_max)
    p = k + d_max
    if n <= 2 * p + 4:
        return (float("nan"), float("nan"), k, d_max, n)
    Yt = Y[p:]
    Xt_blocks = [Y[p - j: -j if j > 0 else None] for j in range(1, p + 1)]
    X = np.column_stack([np.ones(len(Yt))] + Xt_blocks)
    try:
        beta, *_ = np.linalg.lstsq(X, Yt, rcond=None)
    except np.linalg.LinAlgError:
        return (float("nan"), float("nan"), k, d_max, n)
    res = Yt - X @ beta
    sigma_eps = (res.T @ res) / (Yt.shape[0] - X.shape[1])
    XtX_inv = np.linalg.pinv(X.T @ X)

    # Wald for "x_col -> y_col": joint zero on first-k lag coefs of x in y eq.
    # Coefficient layout in beta: rows are regressors (const, lag1_y, lag1_x,
    # lag2_y, lag2_x, ..., lagp_y, lagp_x), columns are equations (y_col, x_col)
    # The order we built corresponds to: const, then for each lag j the
    # vector [y, x] of country.
    R_rows_x_in_y = []
    for j in range(1, k + 1):
        # row index in beta of the lag-j x coefficient
        row_idx = 1 + (j - 1) * 2 + 1  # const + (j-1)*2 + (x is 2nd of pair)
        R = np.zeros((1, X.shape[1]))
        R[0, row_idx] = 1.0
        R_rows_x_in_y.append(R)
    R_x = np.vstack(R_rows_x_in_y)

    # The y-equation coefficients are the first column of beta.
    beta_y = beta[:, 0]
    Rb = R_x @ beta_y
    var_b = sigma_eps[0, 0] * R_x @ XtX_inv @ R_x.T
    try:
        wald = float(Rb @ np.linalg.solve(var_b, Rb))
    except np.linalg.LinAlgError:
        return (float("nan"), float("nan"), k, d_max, n)
    p_val = float(1 - stats.chi2.cdf(wald, df=k))
    return (wald, p_val, k, d_max, n)


def run_ty_panel(
    panel: pd.DataFrame, y_col: str = "gini_disp", x_col: str = "v2x_corr"
) -> pd.DataFrame:
    rows = []
    for iso, g in panel.groupby("iso3"):
        n = g[[y_col, x_col]].dropna().shape[0]
        if n < 25:
            continue
        # x -> y
        wald_xy, p_xy, k_xy, d_xy, n_xy = ty_country(g, y_col, x_col)
        # y -> x (swap)
        wald_yx, p_yx, k_yx, d_yx, n_yx = ty_country(g, x_col, y_col)
        rows.append({
            "iso3": iso,
            "direction": "corr_to_gini",
            "k": k_xy, "d_max": d_xy, "n_obs": n_xy,
            "wald": wald_xy, "p_value": p_xy,
        })
        rows.append({
            "iso3": iso,
            "direction": "gini_to_corr",
            "k": k_yx, "d_max": d_yx, "n_obs": n_yx,
            "wald": wald_yx, "p_value": p_yx,
        })
    return pd.DataFrame(rows)
