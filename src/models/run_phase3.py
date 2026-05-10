"""Phase 3 driver: 6 PVAR-diff specifications × 2 lags."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.models.pvar_diff import (
    PVARSpec,
    companion_eigenvalues,
    fit_pvar_diff,
    flatten_results,
)
from src.utils.paths import PROCESSED, TABLES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


SPECS_BASE = [
    ("S1", ("gini_disp", "v2x_corr")),
    ("S2", ("gini_disp", "v2x_corr", "log_gdp_pc")),
    ("S3", ("gini_disp", "v2x_corr", "education_years")),
    ("S4", ("gini_disp", "v2x_corr", "trade_open")),
    ("S5", ("gini_disp", "v2x_corr", "gov_size")),
    ("S6", ("gini_disp", "v2x_corr", "log_gdp_pc",
            "education_years", "trade_open", "gov_size")),
]


def main() -> None:
    panel = pd.read_parquet(PROCESSED / "panel.parquet")

    all_rows = []
    stab_rows = []
    for lags in (1, 2):
        for label, dep_vars in SPECS_BASE:
            spec = PVARSpec(label=f"{label}_lag{lags}", dep_vars=dep_vars, lags=lags)
            try:
                out = fit_pvar_diff(panel, spec, use_year_fe=True, cluster_entity=True)
            except Exception as e:
                log.error("[%s] FAILED: %s", spec.label, e)
                continue
            all_rows.append(flatten_results(out))
            eig = companion_eigenvalues(out["results"], dep_vars, lags)
            stab_rows.append({
                "spec_lag": spec.label,
                "max_modulus": float(np.max(np.abs(eig))),
                "stable": bool(np.max(np.abs(eig)) < 1.0),
                "n_countries": out["n_countries"],
                "n_obs": out["n_obs"],
            })

    all_coefs = pd.concat(all_rows, ignore_index=True)
    all_coefs.to_csv(TABLES / "main_estimation.csv", index=False)
    pd.DataFrame(stab_rows).to_csv(TABLES / "main_stability.csv", index=False)

    print("=== Stability ===")
    print(pd.DataFrame(stab_rows).to_string(index=False))
    print()
    headline = all_coefs[
        (all_coefs.lags == 1)
        & (all_coefs.equation.isin(["gini_disp", "v2x_corr"]))
        & (all_coefs.regressor.isin([
            "d_gini_disp_l1", "d_v2x_corr_l1",
        ]))
    ].copy()
    headline = headline[headline.spec.str.endswith("_lag1")]
    print("=== Headline (lag = 1) cross-equation coefficients ===")
    print(headline[["spec", "equation", "regressor", "coef", "se", "t", "p", "n_countries"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
