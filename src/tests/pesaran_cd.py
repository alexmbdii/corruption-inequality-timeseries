"""Pesaran (2004, 2015) CD test for cross-sectional dependence.

CD = sqrt(2T / (N(N-1))) * sum_{i<j} rho_ij  ~ N(0,1)

where rho_ij is the pair-wise sample correlation of OLS residuals
e_it from a regression of x_it on a constant and country fixed effects
(equivalently: country-demeaned series).

Also returns the average absolute correlation rho_bar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CDResult:
    statistic: float
    p_value: float
    avg_corr: float
    avg_abs_corr: float
    n_pairs: int
    n_countries: int
    n_periods: int

    def as_dict(self) -> dict:
        return self.__dict__


def cd_test(panel: pd.DataFrame, var: str, demean: str = "country") -> CDResult:
    """Run Pesaran CD test.

    panel: long DataFrame with columns ('iso3', 'year', var).
    demean='country' subtracts country means (FE residual proxy);
    use 'country+year' for two-way FE.
    """
    df = panel[["iso3", "year", var]].dropna().copy()
    if demean == "country":
        df["resid"] = df[var] - df.groupby("iso3")[var].transform("mean")
    elif demean == "country+year":
        df["resid"] = (
            df[var]
            - df.groupby("iso3")[var].transform("mean")
            - df.groupby("year")[var].transform("mean")
            + df[var].mean()
        )
    else:
        df["resid"] = df[var] - df[var].mean()
    wide = df.pivot_table(index="year", columns="iso3", values="resid")
    iso = list(wide.columns)
    N = len(iso); T = wide.shape[0]
    rho_sum = 0.0
    abs_rho_sum = 0.0
    rho_count = 0
    pair_count = 0
    for i in range(N):
        xi = wide.iloc[:, i].values
        for j in range(i + 1, N):
            xj = wide.iloc[:, j].values
            mask = ~np.isnan(xi) & ~np.isnan(xj)
            t_ij = mask.sum()
            if t_ij < 3:
                pair_count += 1
                continue
            xi_, xj_ = xi[mask], xj[mask]
            num = ((xi_ - xi_.mean()) * (xj_ - xj_.mean())).sum()
            den = np.sqrt(((xi_ - xi_.mean()) ** 2).sum() * ((xj_ - xj_.mean()) ** 2).sum())
            if den == 0:
                pair_count += 1
                continue
            rho = num / den
            rho_sum += np.sqrt(t_ij) * rho
            abs_rho_sum += abs(rho)
            rho_count += 1
            pair_count += 1
    cd_stat = np.sqrt(2.0 / (N * (N - 1))) * rho_sum
    p = 2 * (1 - stats.norm.cdf(abs(cd_stat)))
    avg_rho = rho_sum / (rho_count * np.sqrt(T)) if rho_count else np.nan
    avg_abs_rho = abs_rho_sum / rho_count if rho_count else np.nan
    return CDResult(
        statistic=float(cd_stat),
        p_value=float(p),
        avg_corr=float(avg_rho),
        avg_abs_corr=float(avg_abs_rho),
        n_pairs=int(rho_count),
        n_countries=int(N),
        n_periods=int(T),
    )
