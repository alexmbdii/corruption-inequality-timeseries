"""Phase 6 — robustness sweep.

Each spec runs DH (lags 1 and 2, both directions) on the chosen sub-panel.
Bootstrap B = 200 (vs 500 for headline) to keep runtime manageable. Headline
panel is 1990–2023 with v2x_corr × gini_disp; specs vary one dimension at a
time. Output: a single robustness matrix in CSV + LaTeX.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.tests.dh_bootstrap import dh_test
from src.utils.paths import PROCESSED, TABLES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

B = 200


def _signif(p: float) -> str:
    if np.isnan(p): return "—"
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def run_dh_pair(panel: pd.DataFrame, corr_var: str, gini_var: str,
                 spec_label: str, lags=(1, 2), B: int = B,
                 difference: bool = True) -> list[dict]:
    rows = []
    for K in lags:
        for direction, y_col, x_col in [
            ("corr_to_gini", gini_var, corr_var),
            ("gini_to_corr", corr_var, gini_var),
        ]:
            res = dh_test(panel, y_col=y_col, x_col=x_col, K=K,
                          direction_label=direction, B=B, seed=42,
                          difference=difference)
            rows.append({
                "spec": spec_label,
                "corr_var": corr_var,
                "gini_var": gini_var,
                "direction": direction,
                "lags": K,
                "transform": "first-diff" if difference else "levels",
                "Wbar": res.Wbar,
                "Ztilde": res.Ztilde,
                "p_boot": res.Ztilde_p_boot,
                "p_asym": res.Ztilde_p_asym,
                "n_countries": res.n_countries,
            })
    return rows


def main() -> None:
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    rows: list[dict] = []

    # --- 1. V-Dem sub-indices ---
    for corr_alt in ("v2x_pubcorr", "v2x_execorr", "v2x_jucorr_inv"):
        log.info("R1 | %s × gini_disp", corr_alt)
        rows += run_dh_pair(panel, corr_alt, "gini_disp",
                            f"R1_{corr_alt}_x_gini_disp")

    # --- 2. WGI swap (post-2002) ---
    log.info("R2 | wgi_coc_inv × gini_disp (post-2002)")
    sub = panel[panel.year >= 2002].copy()
    rows += run_dh_pair(sub, "wgi_coc_inv", "gini_disp",
                        "R2_wgi_coc_inv_x_gini_disp_post2002")

    # --- 3. gini_mkt swap ---
    log.info("R3 | v2x_corr × gini_mkt")
    rows += run_dh_pair(panel, "v2x_corr", "gini_mkt",
                        "R3_v2x_corr_x_gini_mkt")

    # --- 4. WID top 10% swap ---
    log.info("R4 | v2x_corr × wid_top10")
    rows += run_dh_pair(panel, "v2x_corr", "wid_top10",
                        "R4_v2x_corr_x_wid_top10")

    # --- 5. SWIID 100-imputation propagation ---
    # Strictly, Rubin's rules combine ESTIMATES with their VARIANCES; pooling a
    # test statistic is a non-standard adaptation. We do the closest principled
    # thing for a Wald-style asymptotically-N(0,1) statistic: the within-
    # imputation variance of Ztilde is 1 by construction (asymptotic null),
    # and we add (1 + 1/m) × Var_b(Ztilde) for between-imputation uncertainty,
    # then standardise. Strictly this is "imputation-averaged DH evidence with
    # Rubin-style variance correction", not pure Rubin's rules. Series are
    # first-differenced to comply with the I(1) verdict (Hard Rule 1).
    log.info("R5 | SWIID 100 draws (imputation-averaged DH with Rubin-style "
             "variance correction, differenced)…")
    draws = pd.read_parquet(PROCESSED / "swiid_draws.parquet")
    panel_min = panel[["iso3", "year", "v2x_corr"]].dropna()
    coef_records = []
    n_imp = 100
    for K in (1, 2):
        for direction, y_col, x_col in [
            ("corr_to_gini", "gini_disp", "v2x_corr"),
            ("gini_to_corr", "v2x_corr", "gini_disp"),
        ]:
            stats_per_imp = []
            for m in range(1, n_imp + 1):
                d_m = draws[draws.imp == m][["iso3", "year", "gini_disp", "gini_mkt"]]
                merged = panel_min.merge(d_m, on=["iso3", "year"], how="inner")
                res = dh_test(merged, y_col=y_col, x_col=x_col, K=K,
                              direction_label=direction, B=0, seed=42 + m,
                              difference=True)
                stats_per_imp.append((res.Wbar, res.Ztilde, res.Ztilde_p_asym))
            stats_arr = np.array(stats_per_imp)
            valid = ~np.isnan(stats_arr[:, 1])
            n_valid = int(valid.sum())
            mean_W = float(np.nanmean(stats_arr[:, 0]))
            mean_Z = float(np.nanmean(stats_arr[:, 1]))
            # Rubin's-rules variance correction: total variance =
            # within-imputation + (1+1/m) × between-imputation. For the Ztilde
            # statistic the asymptotic within-imp variance is 1, so total SD is
            # sqrt(1 + (1+1/m) × Var(Ztilde across imp)).
            between_var = float(np.nanvar(stats_arr[valid, 1], ddof=1))
            total_sd = float(np.sqrt(1.0 + (1.0 + 1.0 / n_valid) * between_var))
            from scipy.stats import norm as _norm
            z_pooled = mean_Z / total_sd
            p_pooled = float(1 - _norm.cdf(z_pooled))
            rows.append({
                "spec": "R5_SWIID_imputation_avg",
                "corr_var": "v2x_corr",
                "gini_var": f"gini_disp ({n_valid} imputations)",
                "direction": direction,
                "lags": K,
                "transform": "first-diff",
                "Wbar": mean_W, "Ztilde": mean_Z,
                "p_boot": p_pooled, "p_asym": np.nan,
                "n_countries": 0,
            })
            for m_idx, (w, z, p) in enumerate(stats_per_imp):
                coef_records.append({
                    "direction": direction, "lags": K, "imp": m_idx + 1,
                    "Wbar": w, "Ztilde": z, "p_asym": p,
                })
    pd.DataFrame(coef_records).to_csv(TABLES / "robust_swiid_imputations.csv",
                                       index=False)

    # --- 6. Sub-samples by region/income ---
    region_sets = {
        "OECD": panel[panel.income_group.isin(["HIC"])].copy(),
        "non-OECD": panel[~panel.income_group.isin(["HIC"])].copy(),
        "LAC": panel[panel.region == "LCN"].copy(),
        "SSA": panel[panel.region == "SSF"].copy(),
        "post-Soviet": panel[panel.iso3.isin([
            "ARM", "AZE", "BLR", "EST", "GEO", "KAZ", "KGZ", "LTU", "LVA",
            "MDA", "RUS", "TJK", "TKM", "UKR", "UZB"
        ])].copy(),
    }
    for name, sub in region_sets.items():
        if sub.empty:
            continue
        log.info("R6 | sub-sample %s (N=%d)", name, sub.iso3.nunique())
        rows += run_dh_pair(sub, "v2x_corr", "gini_disp",
                            f"R6_subsample_{name}", lags=(1, 2), B=150)

    # --- 7. Lag sensitivity ---
    log.info("R7 | lag sensitivity (1, 2, 3)")
    rows += run_dh_pair(panel, "v2x_corr", "gini_disp",
                        "R7_lag_sensitivity", lags=(1, 2, 3))

    # --- 8. Period sensitivity ---
    for ymin, ymax, label in [(1995, 2010, "1995-2010"),
                              (2000, 2020, "2000-2020"),
                              (2010, 2023, "2010-2023")]:
        log.info("R8 | period %s", label)
        sub = panel[(panel.year >= ymin) & (panel.year <= ymax)]
        rows += run_dh_pair(sub, "v2x_corr", "gini_disp",
                            f"R8_period_{label}")

    df = pd.DataFrame(rows)
    df["sig"] = df["p_boot"].map(_signif)
    df.to_csv(TABLES / "robustness_matrix.csv", index=False)
    print(df.to_string(index=False))
    print("\nWrote outputs/tables/robustness_matrix.csv")


if __name__ == "__main__":
    main()
