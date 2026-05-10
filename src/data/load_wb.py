"""World Bank data loaders via wbgapi.

Indicators:
  WGI Control of Corruption: CC.EST
  GDP per capita constant USD: NY.GDP.PCAP.KD     -> log_gdp_pc
  Trade openness (% of GDP): NE.TRD.GNFS.ZS       -> trade_open
  Govt consumption (% GDP):  NE.CON.GOVT.ZS       -> gov_size
  Mean years of schooling 15+ (UNESCO UIS): SE.SCH.LIFE  fallback search
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import wbgapi as wb


def _fetch(indicator: str, year_min: int, year_max: int, db: int | None = None) -> pd.DataFrame:
    kw = dict(time=range(year_min, year_max + 1), labels=False, skipBlanks=True, columns="time")
    if db is not None:
        kw["db"] = db
    df = wb.data.DataFrame(indicator, **kw)
    long = df.reset_index().melt(id_vars="economy", var_name="year", value_name=indicator)
    long["year"] = long["year"].astype(str).str.replace("YR", "", regex=False).astype(int)
    long = long.rename(columns={"economy": "iso3"})
    return long


def load_wgi_coc(year_min: int = 1990, year_max: int = 2023) -> pd.DataFrame:
    # WGI was migrated to the "Worldwide Governance Indicators" database (db=3)
    # in 2024; the legacy CC.EST under WDI was archived.
    df = _fetch("GOV_WGI_CC.EST", year_min, year_max, db=3).rename(
        columns={"GOV_WGI_CC.EST": "wgi_coc"}
    )
    df["wgi_coc_inv"] = -df["wgi_coc"]
    return df


def load_wdi_controls(year_min: int = 1990, year_max: int = 2023) -> pd.DataFrame:
    parts = []
    indicators = {
        "NY.GDP.PCAP.KD": "gdp_pc_constUSD",
        "NE.TRD.GNFS.ZS": "trade_open",
        "NE.CON.GOVT.ZS": "gov_size",
    }
    for ind, name in indicators.items():
        d = _fetch(ind, year_min, year_max).rename(columns={ind: name})
        parts.append(d)
    out = parts[0]
    for d in parts[1:]:
        out = out.merge(d, on=["iso3", "year"], how="outer")
    out["log_gdp_pc"] = np.log(out["gdp_pc_constUSD"])
    return out


def load_country_meta() -> pd.DataFrame:
    econ = wb.economy.DataFrame(skipAggs=True).reset_index()
    econ = econ.rename(columns={
        "id": "iso3",
        "name": "country_name",
        "region": "region",
        "incomeLevel": "income_group",
    })
    keep = [c for c in ["iso3", "country_name", "region", "income_group"] if c in econ.columns]
    return econ[keep]


if __name__ == "__main__":
    print("WGI…")
    wgi = load_wgi_coc()
    print(wgi.shape, wgi.head())
    print("WDI…")
    wdi = load_wdi_controls()
    print(wdi.shape, wdi.head())
    print("meta…")
    meta = load_country_meta()
    print(meta.shape, meta.head())
