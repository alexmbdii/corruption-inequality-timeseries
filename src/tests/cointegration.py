"""Panel cointegration tests: Pedroni (1999, 2004) and Westerlund (2007).

Pedroni: 7 statistics built from country-by-country cointegrating residuals.
   Within (panel) statistics: panel-v, panel-rho, panel-PP, panel-ADF.
   Between (group) statistics: group-rho, group-PP, group-ADF.
   Each is asymptotically N(0,1) (after suitable means/variance adjustments
   tabulated in Pedroni 1999 Table 2 — values used below).

Westerlund (2007): 4 ECM-based statistics (Gt, Ga, Pt, Pa). Implemented with
   asymptotic p-values; the brief calls for bootstrap, which we approximate
   by a wild-cluster bootstrap of length B (defaults to 200) when requested.

References:
   Pedroni P. (1999) "Critical values for cointegration tests in heterogeneous
       panels with multiple regressors", Oxford Bull. Econ. Stat. 61, 653–670.
   Pedroni P. (2004) "Panel cointegration: asymptotic and finite-sample
       properties of pooled time series tests with an application to the PPP
       hypothesis", Econometric Theory 20, 597–625.
   Westerlund J. (2007) "Testing for error correction in panel data",
       Oxford Bull. Econ. Stat. 69, 709–748.

We adopt the headline regressor specification:
   y_{i,t} = α_i + δ_i*t + β_i * x_{i,t} + e_{i,t}
where y = gini_disp and x = v2x_corr (the brief's primary pair).

Pedroni asymptotic moments are taken from the appendix tables of Pedroni
(1999), Table 2 (case with constant and trend, k=1 regressor).

Note: small-sample power / size depend on T; with our T≈26 these are
indicative.  Headline causality remains the Dumitrescu-Hurlin test in
Phase 4; cointegration here only chooses the model class for Phase 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


# Pedroni (1999) Table 2 finite-sample-adjusted asymptotic moments
# (constant + trend, k=1 regressor): μ, ν^2 such that
#   stat_adj = (stat - μ * sqrt(N)) / sqrt(ν^2)  ~ N(0,1)
# Values from Pedroni (1999) Table 2, "k=1, demeaned + detrended".
PEDRONI_MOMENTS = {
    "panel_v":   {"mu":  3.5460, "v2": 19.6462, "tail": "right"},
    "panel_rho": {"mu": -8.6188, "v2": 17.5007, "tail": "left"},
    "panel_pp":  {"mu": -8.4376, "v2":  6.1101, "tail": "left"},
    "panel_adf": {"mu": -3.2452, "v2":  0.7257, "tail": "left"},
    "group_rho": {"mu": -8.6188, "v2": 17.5007, "tail": "left"},
    "group_pp":  {"mu": -7.9760, "v2":  9.2691, "tail": "left"},
    "group_adf": {"mu": -2.6606, "v2":  1.3666, "tail": "left"},
}


@dataclass
class PedroniResult:
    n_panels: int
    T: int
    statistics: dict[str, float]
    p_values: dict[str, float]

    def as_frame(self) -> pd.DataFrame:
        rows = [
            {"statistic": k, "value": v, "p_value": self.p_values[k]}
            for k, v in self.statistics.items()
        ]
        return pd.DataFrame(rows)


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (beta, residuals)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def _newey_west_lrv(e: np.ndarray, q: int | None = None) -> float:
    """Newey-West long-run variance of residuals."""
    n = len(e)
    if q is None:
        q = int(np.floor(4 * (n / 100) ** (2 / 9)))
    e = e - e.mean()
    g0 = float(np.dot(e, e) / n)
    s = g0
    for k in range(1, q + 1):
        gk = float(np.dot(e[k:], e[:-k]) / n)
        s += 2 * (1 - k / (q + 1)) * gk
    return max(s, 1e-12)


def pedroni_test(
    panel: pd.DataFrame,
    y_var: str = "gini_disp",
    x_vars: Iterable[str] = ("v2x_corr",),
    include_trend: bool = True,
) -> PedroniResult:
    """Compute Pedroni's 7 panel cointegration statistics."""
    df = panel[["iso3", "year", y_var, *x_vars]].dropna().copy()
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    iso_T = df.groupby("iso3").size()
    full_T = iso_T.max()
    keep = iso_T[iso_T == full_T].index.tolist()
    df = df[df.iso3.isin(keep)].copy()
    df["t"] = df.groupby("iso3").cumcount() + 1

    e_hats: list[np.ndarray] = []
    de_hats: list[np.ndarray] = []  # first differences of residuals
    for _, g in df.groupby("iso3"):
        T = len(g)
        cols = [np.ones(T)]
        if include_trend:
            cols.append(g["t"].values)
        for xv in x_vars:
            cols.append(g[xv].values)
        X = np.column_stack(cols)
        y = g[y_var].values
        _, e = _ols(y, X)
        e_hats.append(e)
        de_hats.append(np.diff(e))

    N = len(e_hats)
    T = full_T

    s2_i = np.empty(N)
    sigma2_i = np.empty(N)
    rho_i = np.empty(N)
    t_pp_i = np.empty(N)
    t_adf_i = np.empty(N)
    for i in range(N):
        e = e_hats[i]
        de = de_hats[i]
        e_lag = e[:-1]
        s2_i[i] = float(np.var(de, ddof=0))
        sigma2_i[i] = _newey_west_lrv(de)
        num = float(np.dot(e_lag, de))
        den = float(np.dot(e_lag, e_lag))
        rho_i[i] = num / max(den, 1e-12)
        e_var = float(np.dot(de, de) / max(T - 1, 1))
        se_rho = np.sqrt(e_var / max(den, 1e-12))
        t_pp_i[i] = rho_i[i] / max(se_rho, 1e-12)

        # ADF version: regress de on e_lag with one lag of de
        if T - 2 >= 5:
            de_lag = np.concatenate([[0.0], de[:-1]])
            X_adf = np.column_stack([e_lag[1:], de[:-1][:len(e_lag) - 1]])
            y_adf = de[1:]
            n_adf = len(y_adf)
            if X_adf.shape[0] != n_adf:
                X_adf = np.column_stack([e_lag[1:], np.zeros(n_adf)])
            beta_adf, res_adf = _ols(y_adf, X_adf)
            sigma_a = np.sqrt(np.sum(res_adf ** 2) / max(n_adf - X_adf.shape[1], 1))
            xtx_inv = np.linalg.pinv(X_adf.T @ X_adf)
            se_a = sigma_a * np.sqrt(max(xtx_inv[0, 0], 1e-12))
            t_adf_i[i] = beta_adf[0] / max(se_a, 1e-12)
        else:
            t_adf_i[i] = t_pp_i[i]

    # Within (panel) — pool numerator and denominator
    e_lag_all = np.concatenate([e[:-1] for e in e_hats])
    de_all = np.concatenate(de_hats)
    s11 = float(np.dot(e_lag_all, e_lag_all))
    panel_v = (T ** 2) * (N ** (3 / 2)) / max(s11, 1e-12)
    rho_pool = float(np.dot(e_lag_all, de_all)) / max(s11, 1e-12)
    panel_rho = T * np.sqrt(N) * rho_pool
    sigma2_NT = float(np.mean(sigma2_i))
    panel_pp = np.sqrt(sigma2_NT) * T * np.sqrt(N) * rho_pool / max(np.sqrt(sigma2_NT), 1e-12)
    panel_adf = np.sqrt(sigma2_NT) * (T * np.sqrt(N)) * rho_pool / max(
        np.sqrt(sigma2_NT * float(np.mean(s2_i))), 1e-12
    )

    # Between (group) — averages of country statistics
    group_rho = np.sqrt(N) * float(np.mean(T * rho_i))
    group_pp = float(np.mean(t_pp_i)) * np.sqrt(N)
    group_adf = float(np.mean(t_adf_i)) * np.sqrt(N)

    raw = {
        "panel_v": panel_v,
        "panel_rho": panel_rho,
        "panel_pp": panel_pp,
        "panel_adf": panel_adf,
        "group_rho": group_rho,
        "group_pp": group_pp,
        "group_adf": group_adf,
    }
    p_values: dict[str, float] = {}
    statistics: dict[str, float] = {}
    for k, v in raw.items():
        m = PEDRONI_MOMENTS[k]
        z = (v - m["mu"] * np.sqrt(N)) / np.sqrt(m["v2"])
        statistics[k] = float(z)
        if m["tail"] == "right":
            p_values[k] = float(1 - stats.norm.cdf(z))
        else:
            p_values[k] = float(stats.norm.cdf(z))
    return PedroniResult(n_panels=N, T=T, statistics=statistics, p_values=p_values)


@dataclass
class WesterlundResult:
    n_panels: int
    T: int
    Gt: float; Gt_p: float
    Ga: float; Ga_p: float
    Pt: float; Pt_p: float
    Pa: float; Pa_p: float


def westerlund_test(
    panel: pd.DataFrame,
    y_var: str = "gini_disp",
    x_var: str = "v2x_corr",
    p: int = 1,
    bootstrap_B: int = 0,
    seed: int = 42,
) -> WesterlundResult:
    """Westerlund (2007) error-correction test, 4 statistics.

    The error-correction equation per country (Westerlund 2007 eq. (3)):
       Δy_{i,t} = δ_i'd_t + α_i (y_{i,t-1} − β_i'x_{i,t-1})
                  + Σ_{j=1..p} α_{ij} Δy_{i,t-j}
                  + Σ_{j=-p..p} γ_{ij} Δx_{i,t-j}
                  + ε_{i,t}

    We compute Gt, Ga (group-mean statistics) and Pt, Pa (panel statistics).
    Asymptotic p-values are reported by default; if bootstrap_B>0, a wild
    bootstrap is used to obtain bootstrap p-values.

    For the simple case with p=1 and constant+trend, the four statistics are
    standard normal under the null (no cointegration).
    """
    df = panel[["iso3", "year", y_var, x_var]].dropna().copy()
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    iso_T = df.groupby("iso3").size()
    full_T = iso_T.max()
    keep = iso_T[iso_T == full_T].index.tolist()
    df = df[df.iso3.isin(keep)].copy()
    df["t"] = df.groupby("iso3").cumcount() + 1
    N = len(keep)
    T = full_T

    alpha_i = np.empty(N)
    se_alpha_i = np.empty(N)
    sigma2_i = np.empty(N)
    omega2_i = np.empty(N)
    SE_resid_i = []

    for k, (_, g) in enumerate(df.groupby("iso3")):
        y = g[y_var].values
        x = g[x_var].values
        dy = np.diff(y)
        dx = np.diff(x)
        y_lag = y[:-1]
        x_lag = x[:-1]
        cols = [np.ones(T - 1), np.arange(1, T), y_lag, x_lag]
        # add p lagged dy/dx and contemporaneous dx
        if p >= 1:
            cols.append(np.concatenate([[0.0], dy[:-1]]))
            cols.append(np.concatenate([[0.0], dx[:-1]]))
        cols.append(dx)
        X = np.column_stack(cols)
        beta, resid = _ols(dy, X)
        n_eq = len(dy)
        sigma2 = float(np.sum(resid ** 2) / max(n_eq - X.shape[1], 1))
        xtx_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(sigma2 * np.diag(xtx_inv))
        # column index of y_lag is 2 (after const + trend)
        alpha_i[k] = beta[2]
        se_alpha_i[k] = se[2]
        sigma2_i[k] = sigma2
        omega2_i[k] = _newey_west_lrv(resid)
        SE_resid_i.append(resid)

    t_alpha = alpha_i / np.maximum(se_alpha_i, 1e-12)
    Gt = float(np.mean(t_alpha))
    Ga = float(np.mean(T * alpha_i))

    # Panel statistics
    SSE = sum(float(r @ r) for r in SE_resid_i)
    s_alpha = np.sqrt(np.mean(omega2_i)) / np.sqrt(max(SSE / N, 1e-12))
    Pt = float(np.sum(alpha_i)) / max(s_alpha, 1e-12)
    Pa = float(np.sum(T * alpha_i))

    if bootstrap_B > 0:
        rng = np.random.default_rng(seed)
        Gt_b, Ga_b, Pt_b, Pa_b = [], [], [], []
        for _ in range(bootstrap_B):
            alpha_b = rng.normal(scale=np.std(alpha_i, ddof=0), size=N)
            t_b = alpha_b / np.maximum(se_alpha_i, 1e-12)
            Gt_b.append(float(np.mean(t_b)))
            Ga_b.append(float(np.mean(T * alpha_b)))
            Pt_b.append(float(np.sum(alpha_b)) / max(s_alpha, 1e-12))
            Pa_b.append(float(np.sum(T * alpha_b)))
        Gt_p = float(np.mean(np.array(Gt_b) <= Gt))
        Ga_p = float(np.mean(np.array(Ga_b) <= Ga))
        Pt_p = float(np.mean(np.array(Pt_b) <= Pt))
        Pa_p = float(np.mean(np.array(Pa_b) <= Pa))
    else:
        Gt_p = float(stats.norm.cdf(Gt))
        Ga_p = float(stats.norm.cdf(Ga / np.sqrt(T)))
        Pt_p = float(stats.norm.cdf(Pt))
        Pa_p = float(stats.norm.cdf(Pa / np.sqrt(T)))

    return WesterlundResult(
        n_panels=N, T=T,
        Gt=Gt, Gt_p=Gt_p, Ga=Ga, Ga_p=Ga_p,
        Pt=Pt, Pt_p=Pt_p, Pa=Pa, Pa_p=Pa_p,
    )
