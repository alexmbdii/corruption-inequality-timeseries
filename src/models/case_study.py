"""Phase 5 — Brazil Lava Jato (2014) case study using synthetic control.

Why Brazil 2014: clean treatment timing (Lava Jato investigation began Mar 2014),
single treated unit (Brazil), Latin-America donor pool. We follow Abadie-Diamond-
Hainmueller / pysyncon-style weights: optimise over weights w on donor countries
to minimise pre-treatment MSPE for the outcome of interest. Two outcomes
estimated separately:
  1. Inequality (gini_disp): does Brazil's inequality diverge from synthetic
     after Lava Jato hits corruption?
  2. Corruption (v2x_corr): direct first-stage check that the shock did
     reduce corruption relative to synthetic.

Pre-treatment window: 2003–2013 (after Lula's election, before Lava Jato).
Post-treatment window: 2014–2020 (Lava Jato peak; data thins after 2020).

Donor pool: Latin American + Caribbean countries (region == 'LCN' in WB
classification) excluding Brazil; require full data 2003–2020 on the
outcome variable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class SCResult:
    treated: str
    outcome: str
    pre_start: int; pre_end: int
    post_end: int
    weights: pd.Series
    gap: pd.Series  # treated - synthetic over time
    pre_mspe: float
    post_avg_gap: float
    avg_pre: float
    avg_post: float


def synthetic_control(
    panel: pd.DataFrame,
    treated: str,
    outcome: str,
    pre_start: int,
    treat_year: int,
    post_end: int,
    donor_region: str = "LCN",
) -> SCResult:
    df = panel[["iso3", "year", "region", outcome]].dropna()
    df = df[(df.year >= pre_start) & (df.year <= post_end)].copy()
    pre_years = list(range(pre_start, treat_year))
    n_pre = len(pre_years)

    # Donor pool: same region, full coverage in pre+post window, exclude treated.
    coverage = df.groupby("iso3").year.nunique()
    full = coverage[coverage == post_end - pre_start + 1].index.tolist()
    region_iso = df[df.region == donor_region].iso3.unique().tolist()
    donors = sorted(set(region_iso) & set(full) - {treated})
    if not donors:
        raise RuntimeError(f"No donors for {treated} in region {donor_region}.")

    # Pivot Y matrix [years x units]
    wide = df.pivot_table(index="year", columns="iso3", values=outcome)
    wide = wide.loc[pre_start:post_end, [treated] + donors].dropna(axis=1)
    donors = [c for c in wide.columns if c != treated]

    Y_pre_t = wide.loc[pre_years, treated].values
    Y_pre_d = wide.loc[pre_years, donors].values
    Y_full_t = wide[treated].values
    Y_full_d = wide[donors].values

    n_d = len(donors)

    def loss(w):
        s = (Y_pre_d @ w) - Y_pre_t
        return float((s ** 2).sum())

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bnds = [(0.0, 1.0)] * n_d
    w0 = np.full(n_d, 1.0 / n_d)
    res = minimize(loss, w0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    w = res.x

    synth = Y_full_d @ w
    gap = pd.Series(Y_full_t - synth, index=wide.index)
    pre_mspe = float(((gap.loc[pre_years]) ** 2).mean())
    post_years = list(range(treat_year, post_end + 1))
    post_avg_gap = float(gap.loc[post_years].mean())

    return SCResult(
        treated=treated,
        outcome=outcome,
        pre_start=pre_start, pre_end=treat_year - 1, post_end=post_end,
        weights=pd.Series(w, index=donors).sort_values(ascending=False),
        gap=gap,
        pre_mspe=pre_mspe,
        post_avg_gap=post_avg_gap,
        avg_pre=float(wide.loc[pre_years, treated].mean()),
        avg_post=float(wide.loc[post_years, treated].mean()),
    )
