"""Phase 2 driver: collects all pre-tests and produces the decision summary."""
from __future__ import annotations

import logging

import pandas as pd

from src.tests.cointegration import pedroni_test, westerlund_test
from src.tests.pesaran_cd import cd_test
from src.utils.paths import PROCESSED, TABLES

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

WINDOW = (1995, 2020)
PRIMARY_PAIR = ("gini_disp", "v2x_corr")


def main() -> None:
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    sub = panel[(panel.year >= WINDOW[0]) & (panel.year <= WINDOW[1])].copy()

    # ---- Pesaran CD (Python implementation, sanity check vs R) ----
    rows = []
    for v in ("v2x_corr", "gini_disp", "log_gdp_pc",
              "education_years", "trade_open", "gov_size", "v2x_polyarchy"):
        if v not in sub.columns:
            continue
        r = cd_test(sub, v, demean="country")
        rows.append({"variable": v, **r.as_dict(), "method": "Pesaran CD (Py)"})
    cd_py = pd.DataFrame(rows)
    cd_py.to_csv(TABLES / "pretests_cd_py.csv", index=False)
    log.info("Wrote pretests_cd_py.csv")
    print("=== Pesaran CD — Python ===")
    print(cd_py.to_string(index=False))
    print()

    # ---- Pedroni cointegration: canonical pco implementation (R) preferred. ----
    pedroni_R_path = TABLES / "pretests_pedroni_R.csv"
    if pedroni_R_path.exists():
        ped_R = pd.read_csv(pedroni_R_path)
        # Use the constant+trend specification as headline.
        ped_R_ic = ped_R[ped_R.spec == "constant+trend"].copy()
        ped_R_ic.to_csv(TABLES / "pretests_pedroni.csv", index=False)
        ped_n_panel = ((ped_R_ic.statistic.isin(["panel_v", "panel_rho",
                                                 "panel_pp", "panel_adf"]))
                       & (ped_R_ic.p_value < 0.05)).sum()
        ped_n_group = ((ped_R_ic.statistic.isin(["group_rho", "group_pp",
                                                 "group_adf"]))
                       & (ped_R_ic.p_value < 0.05)).sum()
        print("=== Pedroni (1999, 2004) via pco — constant+trend ===")
        print(ped_R_ic.to_string(index=False))
        print()
    else:
        # Fallback to Python implementation if pco run hasn't produced output.
        log.warning("Falling back to Python Pedroni (less accurate).")
        ped = pedroni_test(sub, y_var=PRIMARY_PAIR[0], x_vars=(PRIMARY_PAIR[1],),
                           include_trend=True)
        ped_df = ped.as_frame()
        ped_df.to_csv(TABLES / "pretests_pedroni.csv", index=False)
        ped_n_panel = sum(ped.p_values[k] < 0.05
                          for k in ("panel_v", "panel_rho", "panel_pp", "panel_adf"))
        ped_n_group = sum(ped.p_values[k] < 0.05
                          for k in ("group_rho", "group_pp", "group_adf"))

    # ---- Westerlund cointegration (asymptotic + bootstrap) ----
    west = westerlund_test(sub, y_var=PRIMARY_PAIR[0], x_var=PRIMARY_PAIR[1],
                           p=1, bootstrap_B=200, seed=42)
    west_rows = [
        {"statistic": "Gt", "value": west.Gt, "p_value_boot": west.Gt_p},
        {"statistic": "Ga", "value": west.Ga, "p_value_boot": west.Ga_p},
        {"statistic": "Pt", "value": west.Pt, "p_value_boot": west.Pt_p},
        {"statistic": "Pa", "value": west.Pa, "p_value_boot": west.Pa_p},
    ]
    west_df = pd.DataFrame(west_rows)
    west_df["n_panels"] = west.n_panels
    west_df["T"] = west.T
    west_df.to_csv(TABLES / "pretests_westerlund.csv", index=False)
    print("=== Westerlund (2007) — Gt/Ga/Pt/Pa, B=200 wild bootstrap ===")
    print(west_df.to_string(index=False))
    print()

    # ---- Decision ----
    n_panel_reject = int(ped_n_panel)
    n_group_reject = int(ped_n_group)
    west_reject = sum([w["p_value_boot"] < 0.05 for w in west_rows])

    # Engle-Granger residual ADF country share, if produced by R.
    eg_path = TABLES / "pretests_eg_residual_adf.csv"
    if eg_path.exists():
        eg = pd.read_csv(eg_path)
        eg_share = float(eg["reject_5pct"].mean())
        print(f"Engle-Granger residual ADF: {int(eg.reject_5pct.sum())}/{len(eg)} "
              f"countries ({eg_share:.1%}) cointegrated at 5%.\n")
    else:
        eg_share = None

    rule_2of3 = sum([
        n_panel_reject >= 2,   # majority of Pedroni-within (4 stats)
        n_group_reject >= 2,   # majority of Pedroni-between (3 stats)
        west_reject  >= 2,     # majority of Westerlund (4 stats)
    ])
    print("=== Cointegration verdict (rule: 2 of 3 across Pedroni-panel, "
          "Pedroni-group, Westerlund) ===")
    print(f"  Pedroni panel rejects:  {n_panel_reject}/4 at 5%")
    print(f"  Pedroni group rejects:  {n_group_reject}/3 at 5%")
    print(f"  Westerlund rejects:     {west_reject}/4 at 5%")
    cointegrated = rule_2of3 >= 2
    print(f"  COINTEGRATED: {cointegrated}")
    print(f"  → Phase 3 model class: {'Panel VECM' if cointegrated else 'Panel VAR (in differences)'}")

    # Save decision row
    decision = pd.DataFrame([{
        "y": PRIMARY_PAIR[0], "x": PRIMARY_PAIR[1],
        "window_start": WINDOW[0], "window_end": WINDOW[1],
        "pedroni_panel_rejects": int(n_panel_reject),
        "pedroni_group_rejects": int(n_group_reject),
        "westerlund_rejects":    int(west_reject),
        "cointegrated":          bool(cointegrated),
        "phase3_model":          "VECM" if cointegrated else "PVAR_diff",
    }])
    decision.to_csv(TABLES / "phase2_decision.csv", index=False)


if __name__ == "__main__":
    main()
