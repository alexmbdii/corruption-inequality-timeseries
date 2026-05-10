"""Phase 4 driver: DH (asymptotic + bootstrap) + Toda-Yamamoto + pre/post 2008."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.tests.dh_bootstrap import dh_test
from src.tests.toda_yamamoto import run_ty_panel
from src.utils.paths import PROCESSED, TABLES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def run_dh_window(panel: pd.DataFrame, label: str, B: int = 500,
                   difference: bool = True) -> pd.DataFrame:
    rows = []
    for K in (1, 2):
        for direction, y_col, x_col in [
            ("corr_to_gini", "gini_disp", "v2x_corr"),
            ("gini_to_corr", "v2x_corr", "gini_disp"),
        ]:
            res = dh_test(panel, y_col=y_col, x_col=x_col, K=K,
                          direction_label=direction, B=B, seed=42,
                          difference=difference)
            rows.append({
                "window": label,
                "direction": res.direction,
                "lags": res.lags,
                "transform": "first-diff" if difference else "levels",
                "Wbar": res.Wbar,
                "Ztilde": res.Ztilde,
                "p_asym": res.Ztilde_p_asym,
                "p_boot_Ztilde": res.Ztilde_p_boot,
                "p_boot_Wbar": res.Wbar_p_boot,
                "n_countries": res.n_countries,
                "B_eff": res.B,
            })
    return pd.DataFrame(rows)


def main() -> None:
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    log.info("DH on full panel (1990-2023)…")
    dh_full = run_dh_window(panel, "1990-2023", B=500)
    log.info("DH pre-2008 (1990-2007)…")
    dh_pre = run_dh_window(panel[panel.year <= 2007], "1990-2007", B=300)
    log.info("DH post-2008 (2008-2023)…")
    dh_post = run_dh_window(panel[panel.year >= 2008], "2008-2023", B=300)

    dh_all = pd.concat([dh_full, dh_pre, dh_post], ignore_index=True)
    dh_all.to_csv(TABLES / "dh_results.csv", index=False)
    print("=== Dumitrescu-Hurlin ===")
    print(dh_all.to_string(index=False))
    print()

    log.info("Toda-Yamamoto country-by-country (T>=25)…")
    ty = run_ty_panel(panel, y_col="gini_disp", x_col="v2x_corr")
    ty.to_csv(TABLES / "ty_results.csv", index=False)
    summary = (
        ty.groupby("direction")
        .agg(n=("iso3", "count"),
             reject_5=("p_value", lambda s: (s < 0.05).sum()),
             reject_10=("p_value", lambda s: (s < 0.10).sum()),
             reject_1=("p_value", lambda s: (s < 0.01).sum()))
        .reset_index()
    )
    summary.to_csv(TABLES / "ty_summary.csv", index=False)
    print("=== Toda-Yamamoto country-level (T>=25) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
