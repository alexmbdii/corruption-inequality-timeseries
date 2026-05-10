"""SWIID v9.7 loaders.

`load_swiid_summary`: point estimates + SEs for net (gini_disp) and market (gini_mkt) Gini.
`load_swiid_draws`: 100-imputation draws (long format) for uncertainty propagation.
"""
from __future__ import annotations

import pandas as pd

from src.utils.paths import SWIID_DRAWS_DTA, SWIID_SUMMARY_CSV


def load_swiid_summary(year_min: int = 1990, year_max: int = 2023) -> pd.DataFrame:
    df = pd.read_csv(SWIID_SUMMARY_CSV)
    df = df.rename(columns={"country": "country_swiid"})
    keep = [
        "country_swiid", "year",
        "gini_disp", "gini_disp_se",
        "gini_mkt", "gini_mkt_se",
        "abs_red", "abs_red_se", "rel_red", "rel_red_se",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]
    df = df[(df.year >= year_min) & (df.year <= year_max)].copy()
    return df.sort_values(["country_swiid", "year"]).reset_index(drop=True)


def load_swiid_draws(year_min: int = 1990, year_max: int = 2023) -> pd.DataFrame:
    """Return the 100-imputation draws in long format.

    The SWIID v9.x .dta has wide format with `gini_disp1`..`gini_disp100`. We
    melt to long: (country, year, imp, gini_disp, gini_mkt).
    """
    import re
    df = pd.read_stata(str(SWIID_DRAWS_DTA), convert_categoricals=False)
    df = df[(df.year >= year_min) & (df.year <= year_max)].copy()
    pat = re.compile(r"^_(\d+)_gini_(disp|mkt)$")
    disp_cols = [c for c in df.columns if pat.match(c) and pat.match(c).group(2) == "disp"]
    mkt_cols = [c for c in df.columns if pat.match(c) and pat.match(c).group(2) == "mkt"]
    if not disp_cols:
        raise RuntimeError(f"No _N_gini_disp columns in {SWIID_DRAWS_DTA}; cols={list(df.columns)[:30]}")
    id_cols = [c for c in ("country", "year") if c in df.columns]
    long_disp = df[id_cols + disp_cols].melt(
        id_vars=id_cols, value_vars=disp_cols, var_name="_v", value_name="gini_disp"
    )
    long_disp["imp"] = long_disp["_v"].str.extract(r"^_(\d+)_").astype(int)
    long_disp = long_disp.drop(columns="_v")
    long_mkt = df[id_cols + mkt_cols].melt(
        id_vars=id_cols, value_vars=mkt_cols, var_name="_v", value_name="gini_mkt"
    )
    long_mkt["imp"] = long_mkt["_v"].str.extract(r"^_(\d+)_").astype(int)
    long_mkt = long_mkt.drop(columns="_v")
    out = long_disp.merge(long_mkt, on=id_cols + ["imp"], how="outer")
    out = out.rename(columns={"country": "country_swiid"})
    return out.sort_values(["country_swiid", "year", "imp"]).reset_index(drop=True)


if __name__ == "__main__":
    s = load_swiid_summary()
    print("summary:", s.shape, s.columns.tolist())
    print(s.head())
    d = load_swiid_draws()
    print("draws:", d.shape)
    print(d.head())
