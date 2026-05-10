"""Phase 5 driver: panel local projections + Brazil Lava Jato synthetic control."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.case_study import synthetic_control
from src.models.local_projections import lp_irf
from src.utils.paths import FIGURES, PROCESSED, TABLES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    panel = pd.read_parquet(PROCESSED / "panel.parquet")

    # ------------- Local projections -------------
    log.info("LP: corruption shock → inequality")
    lp_c2i = lp_irf(panel, shock_var="v2x_corr", response_var="gini_disp",
                    horizons=range(0, 6),
                    controls=("log_gdp_pc", "v2x_polyarchy"))
    lp_c2i.to_csv(TABLES / "lp_corruption_to_inequality.csv", index=False)
    print("=== LP: shock to v2x_corr → response in gini_disp ===")
    print(lp_c2i.to_string(index=False))
    print()

    log.info("LP: inequality shock → corruption")
    lp_i2c = lp_irf(panel, shock_var="gini_disp", response_var="v2x_corr",
                    horizons=range(0, 6),
                    controls=("log_gdp_pc", "v2x_polyarchy"))
    lp_i2c.to_csv(TABLES / "lp_inequality_to_corruption.csv", index=False)
    print("=== LP: shock to gini_disp → response in v2x_corr ===")
    print(lp_i2c.to_string(index=False))
    print()

    # IRF figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharex=True)
    for ax, (label, df) in zip(axes, [
        ("Corruption shock → inequality (Δgini at h)", lp_c2i),
        ("Inequality shock → corruption (Δv2x_corr at h)", lp_i2c),
    ]):
        ax.plot(df.horizon, df.beta, "-o", color="black")
        ax.fill_between(df.horizon, df.ci_low, df.ci_high, alpha=0.2, color="grey")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("horizon h (years)"); ax.set_title(label)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05_lp_irfs.png", dpi=300)
    fig.savefig(FIGURES / "fig05_lp_irfs.pdf")
    plt.close(fig)

    # ------------- Brazil Lava Jato synthetic control -------------
    log.info("Synthetic control: Brazil Lava Jato 2014, outcome = v2x_corr")
    sc_corr = synthetic_control(panel, treated="BRA", outcome="v2x_corr",
                                pre_start=2003, treat_year=2014,
                                post_end=2020, donor_region="LCN")
    log.info("Synthetic control: Brazil Lava Jato 2014, outcome = gini_disp")
    sc_gini = synthetic_control(panel, treated="BRA", outcome="gini_disp",
                                pre_start=2003, treat_year=2014,
                                post_end=2020, donor_region="LCN")

    # Save weights
    sc_corr.weights.to_csv(TABLES / "case_brazil_weights_corruption.csv",
                           header=["weight"])
    sc_gini.weights.to_csv(TABLES / "case_brazil_weights_inequality.csv",
                           header=["weight"])

    summary = pd.DataFrame([
        {"outcome": "v2x_corr", "treated_avg_pre": sc_corr.avg_pre,
         "treated_avg_post": sc_corr.avg_post,
         "post_avg_gap": sc_corr.post_avg_gap, "pre_mspe": sc_corr.pre_mspe},
        {"outcome": "gini_disp", "treated_avg_pre": sc_gini.avg_pre,
         "treated_avg_post": sc_gini.avg_post,
         "post_avg_gap": sc_gini.post_avg_gap, "pre_mspe": sc_gini.pre_mspe},
    ])
    summary.to_csv(TABLES / "case_brazil_summary.csv", index=False)
    print("=== Brazil Lava Jato synthetic control summary ===")
    print(summary.to_string(index=False))
    print("Top donors (corruption outcome):")
    print(sc_corr.weights.head(5).to_string())
    print()

    # Case-study figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, sc, label in zip(axes,
                             [sc_corr, sc_gini],
                             ["v2x_corr", "gini_disp"]):
        # treated and synthetic levels
        ax.plot(sc.gap.index, sc.gap.values, color="black", linewidth=1.5,
                label="treated − synthetic")
        ax.axvline(2014, linestyle="--", color="red", linewidth=1,
                   label="Lava Jato (2014)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"Brazil − synthetic, {label}")
        ax.set_xlabel("year"); ax.set_ylabel("gap")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05_case_brazil.png", dpi=300)
    fig.savefig(FIGURES / "fig05_case_brazil.pdf")
    plt.close(fig)
    log.info("Phase 5 done.")


if __name__ == "__main__":
    main()
