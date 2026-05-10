"""Course deliverable, section 2: ACF/PACF + ADF/KPSS country-by-country."""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

from src.course.descriptives import REPRESENTATIVE
from src.course.groups import GROUP_COLORS, GROUP_ORDER, add_group
from src.course.style import (CORAL, INK, NAVY, TWO_PANEL, apply_style)
from src.utils.paths import PROCESSED

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*p-value is.*")
warnings.filterwarnings("ignore", message=".*test statistic is outside.*")

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

ALPHA = 0.05
MIN_T = 15
ACF_LAGS = 8


def _load_panel() -> pd.DataFrame:
    p = pd.read_parquet(PROCESSED / "panel.parquet")
    return add_group(p)


def _series(panel: pd.DataFrame, iso: str, var: str) -> np.ndarray:
    s = (panel[panel.iso3 == iso]
         .sort_values("year")[var].dropna().values.astype(float))
    return s


def _safe_adf(s: np.ndarray) -> tuple[float, float] | tuple[None, None]:
    if len(s) < MIN_T:
        return None, None
    try:
        stat, p, *_ = adfuller(s, regression="c", autolag="AIC")
        return float(stat), float(p)
    except Exception:
        return None, None


def _safe_kpss(s: np.ndarray) -> tuple[float, float] | tuple[None, None]:
    if len(s) < MIN_T:
        return None, None
    try:
        stat, p, *_ = kpss(s, regression="c", nlags="auto")
        return float(stat), float(p)
    except Exception:
        return None, None


def _save(fig, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png")
    fig.savefig(FIGURES / f"{stem}.pdf")
    plt.close(fig)


def _restyle_acf_axis(ax, color: str = NAVY) -> None:
    """Re-color the bars produced by statsmodels' plot_acf/plot_pacf to match
    the doc palette and tighten the visual."""
    for line in ax.lines:
        line.set_color(color)
        line.set_markerfacecolor(color)
        line.set_markeredgecolor(color)
    for col in ax.collections:
        col.set_facecolor(color)
        col.set_edgecolor(color)
        col.set_alpha(0.18)


def acf_pacf_grid(panel: pd.DataFrame) -> None:
    isos = REPRESENTATIVE
    with apply_style():
        fig, axes = plt.subplots(4, len(isos), figsize=(6.3, 6.0), sharex=True)
        for j, iso in enumerate(isos):
            for i, var in enumerate(["v2x_corr", "gini_disp"]):
                s = _series(panel, iso, var)
                color = GROUP_COLORS[panel.loc[panel.iso3 == iso, "group"].iloc[0]]
                row_acf, row_pacf = 2 * i, 2 * i + 1
                if len(s) < 5:
                    axes[row_acf, j].set_visible(False)
                    axes[row_pacf, j].set_visible(False)
                    continue
                k = min(ACF_LAGS, len(s) // 2 - 1)
                plot_acf(s, lags=k, ax=axes[row_acf, j])
                plot_pacf(s, lags=k, ax=axes[row_pacf, j], method="ywm")
                _restyle_acf_axis(axes[row_acf, j], color=color)
                _restyle_acf_axis(axes[row_pacf, j], color=color)
                axes[row_acf, j].set_title("")
                axes[row_pacf, j].set_title("")
                axes[row_acf, j].tick_params(labelsize=7)
                axes[row_pacf, j].tick_params(labelsize=7)
                if j == 0:
                    label = "C" if var == "v2x_corr" else "I"
                    axes[row_acf, j].set_ylabel(f"ACF({label})", fontsize=8)
                    axes[row_pacf, j].set_ylabel(f"PACF({label})", fontsize=8)
                if i == 0 and j == 0:
                    pass
                if row_acf == 0:
                    axes[0, j].annotate(iso, xy=(0.5, 1.10),
                                         xycoords="axes fraction",
                                         ha="center", fontsize=10, color=INK)
                if row_pacf == 3:
                    axes[3, j].set_xlabel("Lag", fontsize=8)
        fig.suptitle("ACF et PACF — corruption (rangées 1–2) et inégalités (rangées 3–4)",
                     fontsize=10.5, y=1.00)
        fig.tight_layout()
        _save(fig, "course_acfpacf")


def adf_kpss_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iso, sub in panel.groupby("iso3"):
        group = sub.group.iloc[0]
        out = {"iso3": iso, "group": group}
        for var in ["v2x_corr", "gini_disp"]:
            s = _series(panel, iso, var)
            adf_stat, adf_p = _safe_adf(s)
            kp_stat, kp_p = _safe_kpss(s)
            out[f"{var}_T"] = len(s)
            out[f"{var}_adf_p"] = adf_p
            out[f"{var}_kpss_p"] = kp_p
            if adf_p is None or kp_p is None:
                v = "n/a"
            elif adf_p < ALPHA and kp_p >= ALPHA:
                v = "stationary"
            elif adf_p >= ALPHA and kp_p < ALPHA:
                v = "non-stationary"
            else:
                v = "ambiguous"
            out[f"{var}_verdict"] = v
        rows.append(out)
    df = pd.DataFrame(rows).sort_values("iso3").reset_index(drop=True)
    df.to_csv(TABLES / "course_adf_kpss.csv", index=False)

    summary_rows = []
    for var in ["v2x_corr", "gini_disp"]:
        for g in GROUP_ORDER + ["TOTAL"]:
            sub = df if g == "TOTAL" else df[df.group == g]
            verdicts = sub[f"{var}_verdict"].value_counts().to_dict()
            summary_rows.append({
                "variable": var, "group": g, "n_countries": len(sub),
                "stationary": verdicts.get("stationary", 0),
                "non_stationary": verdicts.get("non-stationary", 0),
                "ambiguous": verdicts.get("ambiguous", 0),
                "n_a": verdicts.get("n/a", 0),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES / "course_adf_kpss_summary.csv", index=False)
    return df


def adf_pvalue_hist(adf_table: pd.DataFrame) -> None:
    with apply_style():
        fig, axes = plt.subplots(1, 2, figsize=TWO_PANEL)
        for ax, var, label in [(axes[0], "v2x_corr", r"Corruption ($v2x_{\mathrm{corr}}$)"),
                               (axes[1], "gini_disp", r"Gini disponible ($gini_{\mathrm{disp}}$)")]:
            ps = adf_table[f"{var}_adf_p"].dropna()
            ax.hist(ps, bins=20, color=NAVY, edgecolor="white", linewidth=0.4, alpha=0.85)
            ax.axvline(ALPHA, color=CORAL, ls="--", lw=1.0,
                       label=fr"$\alpha = {ALPHA}$")
            n_stat = (ps < ALPHA).sum()
            ax.set_title(f"{label}\n{n_stat}/{len(ps)} pays stationnaires (ADF $p<\\alpha$)",
                         fontsize=10)
            ax.set_xlabel(r"$p$-value ADF (niveaux)")
            ax.set_ylabel("Nombre de pays")
            ax.legend()
        fig.tight_layout()
        _save(fig, "course_adf_pvalue_hist")


def acf_diff_compare(panel: pd.DataFrame) -> None:
    isos = REPRESENTATIVE
    with apply_style():
        fig, axes = plt.subplots(2, len(isos), figsize=(6.3, 3.4), sharex=True)
        for j, iso in enumerate(isos):
            color = GROUP_COLORS[panel.loc[panel.iso3 == iso, "group"].iloc[0]]
            s = _series(panel, iso, "v2x_corr")
            if len(s) < 6:
                axes[0, j].set_visible(False)
                axes[1, j].set_visible(False)
                continue
            k = min(ACF_LAGS, len(s) // 2 - 1)
            plot_acf(s, lags=k, ax=axes[0, j])
            d = np.diff(s)
            plot_acf(d, lags=min(ACF_LAGS, len(d) // 2 - 1), ax=axes[1, j])
            for ax in (axes[0, j], axes[1, j]):
                _restyle_acf_axis(ax, color=color)
                ax.set_title("")
                ax.tick_params(labelsize=7)
            axes[0, j].annotate(iso, xy=(0.5, 1.08), xycoords="axes fraction",
                                 ha="center", fontsize=9.5, color=INK)
            axes[1, j].set_xlabel("Lag", fontsize=8)
            if j == 0:
                axes[0, j].set_ylabel("ACF (niveaux)", fontsize=8)
                axes[1, j].set_ylabel(r"ACF ($\Delta$)", fontsize=8)
        fig.suptitle(r"ACF avant/après différenciation — corruption ($v2x_{\mathrm{corr}}$)",
                     fontsize=10.5, y=1.02)
        fig.tight_layout()
        _save(fig, "course_acf_diff_compare")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    panel = _load_panel()
    print("ACF/PACF grid…")
    acf_pacf_grid(panel)
    print("ADF + KPSS country-by-country…")
    adf = adf_kpss_table(panel)
    print("ADF p-value histogram…")
    adf_pvalue_hist(adf)
    print("ACF before/after differencing…")
    acf_diff_compare(panel)
    print("course series_properties: done")


if __name__ == "__main__":
    main()
