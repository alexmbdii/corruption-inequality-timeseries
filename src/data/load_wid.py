"""WID loader.

The WID bulk download is one CSV per country. We stream-extract the entries we
need and filter to the variable/percentile/age/pop combination of interest.

For top 10% pre-tax national income share: variable=sptinc992j,
percentile=p90p100, age=992, pop=j (equal-split adults). Returns a tidy
country-year DataFrame.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd

from src.utils.paths import WID_ZIP

USECOLS = ["country", "variable", "percentile", "year", "value", "age", "pop"]


def load_wid_top10(
    year_min: int = 1990,
    year_max: int = 2023,
    variable: str = "sptincj992",  # pre-tax national income share, top 10% group
    percentile: str = "p90p100",
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    with zipfile.ZipFile(WID_ZIP) as zf:
        names = [
            n for n in zf.namelist()
            if n.startswith("WID_data_") and n.endswith(".csv")
        ]
        for n in names:
            with zf.open(n) as fh:
                try:
                    chunk = pd.read_csv(
                        io.TextIOWrapper(fh, encoding="utf-8"),
                        sep=";",
                        usecols=USECOLS,
                        dtype={"country": "string", "variable": "string",
                               "percentile": "string", "age": "string", "pop": "string"},
                        low_memory=False,
                    )
                except Exception:
                    continue
            mask = (
                (chunk.variable == variable)
                & (chunk.percentile == percentile)
                & (chunk.year >= year_min)
                & (chunk.year <= year_max)
            )
            if mask.any():
                rows.append(chunk.loc[mask, ["country", "year", "value"]].copy())
    if not rows:
        return pd.DataFrame(columns=["country_wid_alpha2", "year", "wid_top10"])
    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={"country": "country_wid_alpha2", "value": "wid_top10"})
    out["wid_top10"] = pd.to_numeric(out["wid_top10"], errors="coerce")
    return out.dropna(subset=["wid_top10"]).sort_values(["country_wid_alpha2", "year"]).reset_index(drop=True)


if __name__ == "__main__":
    df = load_wid_top10()
    print(df.shape)
    print(df.head())
    print("alpha2 codes:", df.country_wid_alpha2.nunique())
