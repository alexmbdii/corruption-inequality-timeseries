"""Barro-Lee MF15+ education loader.

Source: barro-lee-bl2012_MF1599_v2.2.csv (BL v2.2 release). Columns include
WBcode (ISO3), country, year, sex, agefrom, ageto, yr_sch (mean years of
schooling, age 15+, both sexes when sex=='MF').

Keeps MF + 15-999 + 1990-2010, returns ISO3, year, education_years.
Linear-interpolates within each country to annual frequency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.paths import BARRO_LEE_CSV


def load_barro_lee_5y() -> pd.DataFrame:
    df = pd.read_csv(BARRO_LEE_CSV)
    df = df[(df.sex == "MF") & (df.agefrom == 15) & (df.ageto == 999)].copy()
    df = df.rename(columns={"WBcode": "iso3"})
    df = df[["iso3", "country", "year", "yr_sch"]].rename(columns={"yr_sch": "education_years"})
    return df.sort_values(["iso3", "year"]).reset_index(drop=True)


def expand_to_annual(
    df_5y: pd.DataFrame, year_min: int = 1990, year_max: int = 2023
) -> pd.DataFrame:
    """Linear interpolation within country to annual.

    For 1990–2010, BL gives observations every 5 years (1990,1995,...,2010).
    Linear-interpolate the gaps. After 2010, hold the 2010 value constant
    (documented choice; UNESCO UIS extension is left for robustness).
    """
    out = []
    for iso3, g in df_5y.groupby("iso3"):
        g = g[(g.year >= 1985) & (g.year <= 2010)]
        if g.empty:
            continue
        country = g.country.iloc[0]
        full_years = pd.DataFrame({"year": range(year_min, year_max + 1)})
        full_years["iso3"] = iso3
        full_years["country"] = country
        merged = full_years.merge(g[["year", "education_years"]], on="year", how="left")
        merged["education_years"] = merged["education_years"].interpolate(
            method="linear", limit_direction="both"
        )
        last_obs_year = int(g.year.max())
        last_value = float(g.loc[g.year == last_obs_year, "education_years"].iloc[0])
        merged.loc[merged.year > last_obs_year, "education_years"] = last_value
        out.append(merged)
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["iso3", "year", "education_years"])
    return res.sort_values(["iso3", "year"]).reset_index(drop=True)


if __name__ == "__main__":
    five = load_barro_lee_5y()
    print("5y:", five.shape, "countries:", five.iso3.nunique())
    annual = expand_to_annual(five)
    print("annual:", annual.shape, "countries:", annual.iso3.nunique())
    print(annual.head())
