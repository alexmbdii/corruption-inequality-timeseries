"""Phase 1 assembly: produces data/processed/{panel.parquet, swiid_draws.parquet}.

Steps:
  1. Load V-Dem (ISO3 = country_text_id).
  2. Load SWIID summary (country = English name) and harmonize to ISO3.
  3. Load Barro-Lee 5y → annual (ISO3 = WBcode).
  4. WGI + WDI + country meta via wbgapi (ISO3).
  5. WID top 10% (alpha2) → ISO3.
  6. Merge on (iso3, year). Apply panel rule (>=20 years on both v2x_corr & gini_disp).
  7. Save panel.parquet.
  8. Load SWIID draws, filter to panel ISO3s, save swiid_draws.parquet.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry

from src.data.load_barro_lee import expand_to_annual, load_barro_lee_5y
from src.data.load_swiid import load_swiid_draws, load_swiid_summary
from src.data.load_vdem import load_vdem
from src.data.load_wb import load_country_meta, load_wdi_controls, load_wgi_coc
from src.data.load_wid import load_wid_top10
from src.utils.paths import INTERIM, PROCESSED

YEAR_MIN, YEAR_MAX = 1990, 2023
PANEL_RULE_MIN_YEARS = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def _name_to_iso3_swiid(name: str) -> str | None:
    """SWIID uses English names; map to ISO3 via pycountry."""
    if not isinstance(name, str):
        return None
    try:
        c = pycountry.countries.lookup(name)
        return c.alpha_3
    except LookupError:
        manual = {
            "Czech Republic": "CZE", "Czechia": "CZE",
            "Korea": "KOR", "Korea, Republic of": "KOR",
            "Korea, Democratic People's Republic of": "PRK",
            "Russia": "RUS", "Slovak Republic": "SVK",
            "Iran": "IRN", "Vietnam": "VNM", "Viet Nam": "VNM",
            "Laos": "LAO", "Tanzania": "TZA", "Bolivia": "BOL",
            "Venezuela": "VEN", "Moldova": "MDA",
            "Macedonia": "MKD", "North Macedonia": "MKD",
            "Cape Verde": "CPV", "Cabo Verde": "CPV",
            "Ivory Coast": "CIV", "Cote d'Ivoire": "CIV",
            "Congo, Republic of": "COG", "Congo-Brazzaville": "COG",
            "Congo, Democratic Republic of": "COD", "Congo-Kinshasa": "COD",
            "Hong Kong": "HKG", "Macau": "MAC", "Taiwan": "TWN",
            "Brunei": "BRN", "Syria": "SYR",
            "Palestine": "PSE", "Palestinian Territories": "PSE",
            "Kosovo": "XKX", "Yemen": "YEM",
            "East Timor": "TLS", "Timor-Leste": "TLS",
            "Micronesia": "FSM",
            "Eswatini": "SWZ", "Swaziland": "SWZ",
            "Turkey": "TUR", "Türkiye": "TUR",
            "St. Kitts and Nevis": "KNA",
            "St. Lucia": "LCA",
            "St. Vincent and Grenadines": "VCT",
            "São Tomé and Príncipe": "STP",
            "Soviet Union": None, "Yugoslavia": None,
        }
        return manual.get(name)


def _alpha2_to_iso3(a2: str) -> str | None:
    if not isinstance(a2, str) or len(a2) != 2:
        return None
    try:
        return pycountry.countries.get(alpha_2=a2).alpha_3
    except (LookupError, AttributeError):
        return None


def assemble() -> pd.DataFrame:
    log.info("Loading V-Dem…")
    vdem = load_vdem(YEAR_MIN, YEAR_MAX)
    log.info("V-Dem: %s", vdem.shape)
    vdem.to_parquet(INTERIM / "vdem.parquet", index=False)

    log.info("Loading SWIID summary…")
    swiid = load_swiid_summary(YEAR_MIN, YEAR_MAX)
    swiid["iso3"] = swiid["country_swiid"].map(_name_to_iso3_swiid)
    bad = swiid[swiid.iso3.isna()].country_swiid.unique()
    if len(bad):
        log.warning("SWIID names without ISO3 mapping: %s", list(bad)[:30])
    swiid = swiid.dropna(subset=["iso3"]).drop(columns="country_swiid")
    log.info("SWIID summary: %s, %d countries", swiid.shape, swiid.iso3.nunique())
    swiid.to_parquet(INTERIM / "swiid_summary.parquet", index=False)

    log.info("Loading Barro-Lee + interpolating to annual…")
    bl = load_barro_lee_5y()
    bl_annual = expand_to_annual(bl, YEAR_MIN, YEAR_MAX)
    log.info("Barro-Lee annual: %s, %d countries", bl_annual.shape, bl_annual.iso3.nunique())
    bl_annual.to_parquet(INTERIM / "barro_lee_annual.parquet", index=False)

    log.info("Loading WGI Control of Corruption…")
    wgi = load_wgi_coc(YEAR_MIN, YEAR_MAX)
    wgi.to_parquet(INTERIM / "wgi.parquet", index=False)

    log.info("Loading WDI controls…")
    wdi = load_wdi_controls(YEAR_MIN, YEAR_MAX)
    wdi.to_parquet(INTERIM / "wdi.parquet", index=False)

    log.info("Loading country metadata…")
    meta = load_country_meta()
    meta.to_parquet(INTERIM / "country_meta.parquet", index=False)

    log.info("Loading WID top 10%…")
    wid = load_wid_top10(YEAR_MIN, YEAR_MAX)
    wid["iso3"] = wid["country_wid_alpha2"].map(_alpha2_to_iso3)
    wid = wid.dropna(subset=["iso3"]).drop(columns="country_wid_alpha2")
    log.info("WID: %s, %d countries", wid.shape, wid.iso3.nunique())
    wid.to_parquet(INTERIM / "wid_top10.parquet", index=False)

    log.info("Merging panel…")
    panel = vdem.merge(swiid, on=["iso3", "year"], how="outer")
    panel = panel.merge(bl_annual.drop(columns=["country"], errors="ignore"), on=["iso3", "year"], how="left")
    panel = panel.merge(wgi, on=["iso3", "year"], how="left")
    panel = panel.merge(wdi, on=["iso3", "year"], how="left")
    panel = panel.merge(wid, on=["iso3", "year"], how="left")
    panel = panel.merge(meta, on="iso3", how="left")

    obs_corr = panel.dropna(subset=["v2x_corr"]).groupby("iso3").year.nunique()
    obs_gini = panel.dropna(subset=["gini_disp"]).groupby("iso3").year.nunique()
    keep_iso3 = sorted(set(obs_corr[obs_corr >= PANEL_RULE_MIN_YEARS].index)
                       & set(obs_gini[obs_gini >= PANEL_RULE_MIN_YEARS].index))
    log.info("Panel rule (>=%d yrs both): %d/%d countries kept",
             PANEL_RULE_MIN_YEARS, len(keep_iso3), panel.iso3.nunique())

    panel = panel[panel.iso3.isin(keep_iso3)].copy()
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)

    cols_first = ["iso3", "country_name", "country_vdem", "year", "region", "income_group"]
    other = [c for c in panel.columns if c not in cols_first]
    panel = panel[[c for c in cols_first if c in panel.columns] + other]

    panel.to_parquet(PROCESSED / "panel.parquet", index=False)
    log.info("Wrote %s rows=%d cols=%d", PROCESSED / "panel.parquet", *panel.shape)
    return panel


def assemble_swiid_draws(keep_iso3: list[str]) -> pd.DataFrame:
    log.info("Loading SWIID 100-imputation draws…")
    draws = load_swiid_draws(YEAR_MIN, YEAR_MAX)
    draws["iso3"] = draws["country_swiid"].map(_name_to_iso3_swiid)
    draws = draws.dropna(subset=["iso3"])
    draws = draws[draws.iso3.isin(keep_iso3)].drop(columns="country_swiid")
    draws = draws[["iso3", "year", "imp", "gini_disp", "gini_mkt"]]
    draws.to_parquet(PROCESSED / "swiid_draws.parquet", index=False)
    log.info("Wrote %s rows=%d", PROCESSED / "swiid_draws.parquet", draws.shape[0])
    return draws


if __name__ == "__main__":
    panel = assemble()
    keep_iso3 = sorted(panel.iso3.unique().tolist())
    assemble_swiid_draws(keep_iso3)
