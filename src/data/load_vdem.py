"""V-Dem v16 country-year loader.

Keeps only the columns we need across the full project:
  - corruption indices (primary + sub-indices) with codelow/codehigh
  - polyarchy
  - identifiers (country_text_id, year)
"""
from __future__ import annotations

import pandas as pd

from src.utils.paths import VDEM_CSV

# Note on sign conventions:
#   v2x_corr, v2x_pubcorr, v2x_execorr  → higher = MORE corrupt (V-Dem indices)
#   v2jucorrdc                          → higher = LESS corrupt (raw indicator);
#                                         we invert to v2x_jucorr_inv so that
#                                         higher = MORE corrupt for consistency.
# The brief's `v2x_jucorrdc` does not exist in V-Dem v16; the underlying
# judicial-corruption indicator is `v2jucorrdc`.
VDEM_KEEP = [
    "country_text_id", "country_name", "year",
    "v2x_corr", "v2x_corr_codelow", "v2x_corr_codehigh",
    "v2x_pubcorr", "v2x_pubcorr_codelow", "v2x_pubcorr_codehigh",
    "v2x_execorr", "v2x_execorr_codelow", "v2x_execorr_codehigh",
    "v2jucorrdc", "v2jucorrdc_codelow", "v2jucorrdc_codehigh",
    "v2x_polyarchy", "v2x_polyarchy_codelow", "v2x_polyarchy_codehigh",
]


def load_vdem(year_min: int = 1990, year_max: int = 2023) -> pd.DataFrame:
    df = pd.read_csv(VDEM_CSV, usecols=VDEM_KEEP, low_memory=False)
    df = df.rename(columns={"country_text_id": "iso3", "country_name": "country_vdem"})
    df = df[(df.year >= year_min) & (df.year <= year_max)].copy()
    # Invert judicial corruption so higher = more corrupt (matches v2x_corr).
    for c in ("v2jucorrdc", "v2jucorrdc_codelow", "v2jucorrdc_codehigh"):
        df[c.replace("v2jucorrdc", "v2x_jucorr_inv")] = -df[c]
    df = df.drop(columns=["v2jucorrdc", "v2jucorrdc_codelow", "v2jucorrdc_codehigh"])
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_vdem()
    print(df.shape)
    print(df.head())
    print("countries:", df.iso3.nunique(), "years:", df.year.min(), "-", df.year.max())
