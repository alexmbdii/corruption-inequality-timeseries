"""Phase 1 deliverables: appendix table, missingness heatmap, exploratory plots."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.paths import FIGURES, PROCESSED, TABLES

log = logging.getLogger(__name__)

VAR_META = {
    "v2x_corr":          ("V-Dem v16",                "Political corruption index (primary corruption)",
                          "Higher = more corrupt; aggregated over 4 sub-indicators."),
    "v2x_pubcorr":       ("V-Dem v16",                "Public-sector corruption index (robustness)",       ""),
    "v2x_execorr":       ("V-Dem v16",                "Executive corruption index (robustness)",            ""),
    "v2x_jucorr_inv":    ("V-Dem v16",                "Judicial corruption (inverted from v2jucorrdc)",
                          "Higher = more corrupt after sign-flip."),
    "v2x_polyarchy":     ("V-Dem v16",                "Electoral democracy index (control)",                ""),
    "gini_disp":         ("SWIID v9.7 summary",        "Net Gini, post-tax/transfer (primary inequality)",   ""),
    "gini_disp_se":      ("SWIID v9.7 summary",        "SE of net Gini",                                     ""),
    "gini_mkt":          ("SWIID v9.7 summary",        "Market Gini, pre-tax/transfer (robustness)",         ""),
    "gini_mkt_se":       ("SWIID v9.7 summary",        "SE of market Gini",                                  ""),
    "wgi_coc":           ("WB WGI db=3",               "Control of Corruption (1996+ only)",
                          "Higher = better governance; we report wgi_coc_inv for sign-consistency."),
    "wid_top10":         ("WID v2024",                 "Top-10% pre-tax national income share (sptincj992)", ""),
    "log_gdp_pc":        ("WB WDI",                    "ln(GDP per capita, constant USD) — control",         ""),
    "education_years":   ("Barro-Lee v2.2",            "Mean years schooling, 15+ (5y → annual interp.)",
                          "Pre-2010: linear interp. of 1990,1995,...,2010. 2011+: held constant at 2010 (UNESCO UIS extension reserved for robustness)."),
    "trade_open":        ("WB WDI",                    "Trade (% of GDP) — control",                         ""),
    "gov_size":          ("WB WDI",                    "Govt consumption (% of GDP) — control",              ""),
}


def appendix_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_total = len(panel)
    for v, (src, desc, transform) in VAR_META.items():
        if v not in panel.columns:
            continue
        s = panel[v]
        miss = s.isna().mean()
        n_obs = int(s.notna().sum())
        n_country = int(panel.loc[s.notna(), "iso3"].nunique())
        ymin = panel.loc[s.notna(), "year"].min()
        ymax = panel.loc[s.notna(), "year"].max()
        rows.append({
            "variable": v,
            "source": src,
            "description": desc,
            "transformations": transform,
            "n_obs": n_obs,
            "n_countries": n_country,
            "coverage": f"{int(ymin)}–{int(ymax)}" if pd.notna(ymin) else "",
            "missingness": f"{miss:.1%}",
        })
    return pd.DataFrame(rows)


def missingness_heatmap(panel: pd.DataFrame, var: str, fname_stem: str) -> Path:
    pivot = panel.assign(_present=panel[var].notna().astype(int)).pivot_table(
        index="iso3", columns="year", values="_present", aggfunc="max", fill_value=0
    )
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(10, max(8, 0.06 * len(pivot))))
    sns.heatmap(pivot, cbar=False, cmap="Greys", linewidths=0, ax=ax)
    ax.set_title(f"Coverage by country × year for {var} (black = present)")
    ax.set_xlabel("year"); ax.set_ylabel("ISO3")
    ax.tick_params(axis="y", labelsize=5)
    fig.tight_layout()
    p_png = FIGURES / f"{fname_stem}.png"
    fig.savefig(p_png, dpi=300)
    fig.savefig(FIGURES / f"{fname_stem}.pdf")
    plt.close(fig)
    return p_png


def country_mean_within_var_plot(panel: pd.DataFrame, var: str, fname_stem: str) -> Path:
    g = panel.dropna(subset=[var]).groupby("iso3")[var].agg(["mean", "var", "count"])
    g = g[g["count"] >= 10]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(g["mean"], np.sqrt(g["var"]), s=10, alpha=0.6)
    ax.set_xlabel(f"country-mean {var}")
    ax.set_ylabel(f"within-country sd of {var}")
    ax.set_title(f"Between vs within variation: {var} (n={len(g)} countries)")
    fig.tight_layout()
    p_png = FIGURES / f"{fname_stem}.png"
    fig.savefig(p_png, dpi=300)
    fig.savefig(FIGURES / f"{fname_stem}.pdf")
    plt.close(fig)
    return p_png


def run_all() -> None:
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    log.info("panel loaded: %s, countries=%d", panel.shape, panel.iso3.nunique())

    app = appendix_table(panel)
    app.to_csv(TABLES / "data_appendix.csv", index=False)
    with open(TABLES / "data_appendix.tex", "w") as f:
        f.write(app.to_latex(index=False, escape=True))

    missingness_heatmap(panel, "v2x_corr", "fig01_coverage_v2x_corr")
    missingness_heatmap(panel, "gini_disp", "fig01_coverage_gini_disp")

    country_mean_within_var_plot(panel, "v2x_corr", "fig01_betweenwithin_v2x_corr")
    country_mean_within_var_plot(panel, "gini_disp", "fig01_betweenwithin_gini_disp")

    print("DATA APPENDIX")
    print(app.to_string(index=False))
    print()
    print(f"countries: {panel.iso3.nunique()}")
    print(f"years: {panel.year.min()}–{panel.year.max()}")
    print(f"rows: {len(panel)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_all()
