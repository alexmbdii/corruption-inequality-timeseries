"""Course deliverable, section 4: comparisons across OECD / Émergents / Afrique."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from src.course.granger_country import CAT_COLORS, CAT_ORDER
from src.course.groups import GROUP_COLORS, GROUP_ORDER, add_group
from src.course.style import (FULL_PAGE, INK, TWO_PANEL, apply_style)
from src.utils.paths import PROCESSED

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

ACF_LAGS = 8


def _save(fig, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def barplot_categories_by_group() -> None:
    g = pd.read_csv(TABLES / "course_granger_country.csv")
    sub = g[g.category.isin(CAT_ORDER)]
    pivot = (sub.groupby(["group", "category"]).iso3.count()
             .unstack("category").reindex(GROUP_ORDER)[CAT_ORDER].fillna(0))

    with apply_style():
        fig, ax = plt.subplots(figsize=FULL_PAGE)
        bottoms = np.zeros(len(pivot))
        x = np.arange(len(pivot))
        for cat in CAT_ORDER:
            vals = pivot[cat].values
            ax.bar(x, vals, bottom=bottoms, color=CAT_COLORS[cat], label=cat,
                   edgecolor="white", linewidth=1.0, width=0.55)
            for i, v in enumerate(vals):
                if v > 0:
                    txt_color = "white" if cat in ("C→I", "I→C", "Bidir") else INK
                    ax.text(i, bottoms[i] + v / 2, f"{int(v)}",
                            ha="center", va="center", fontsize=9.5,
                            color=txt_color, fontweight="bold")
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n($n={int(pivot.loc[g].sum())}$)"
                            for g in pivot.index], fontsize=10)
        ax.set_ylabel("Nombre de pays")
        ax.set_title("Répartition des catégories causales Granger par groupe")
        ax.legend(title="Direction", loc="upper right",
                  bbox_to_anchor=(1.22, 1.0), title_fontsize=9)
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        _save(fig, "course_barplot_categories_by_group")


def acf_by_group() -> None:
    panel = add_group(pd.read_parquet(PROCESSED / "panel.parquet"))
    with apply_style():
        fig, axes = plt.subplots(1, 2, figsize=TWO_PANEL)
        for ax, var, label in [(axes[0], "v2x_corr", r"Corruption ($v2x_{\mathrm{corr}}$)"),
                                (axes[1], "gini_disp", r"Gini disponible ($gini_{\mathrm{disp}}$)")]:
            for grp in GROUP_ORDER:
                isos = panel.loc[panel.group == grp, "iso3"].unique()
                acf_mat = []
                for iso in isos:
                    s = (panel[panel.iso3 == iso].sort_values("year")[var]
                         .dropna().values.astype(float))
                    if len(s) < ACF_LAGS + 2:
                        continue
                    acf_mat.append(acf(s, nlags=ACF_LAGS, fft=False))
                if not acf_mat:
                    continue
                arr = np.array(acf_mat)
                mean = arr.mean(axis=0)
                ax.plot(range(ACF_LAGS + 1), mean, marker="o", markersize=4,
                        color=GROUP_COLORS[grp],
                        label=fr"{grp} ($n={len(arr)}$)", lw=1.4)
            ax.axhline(0, color=INK, lw=0.5)
            ax.set_xlabel("Lag (années)")
            ax.set_ylabel("ACF moyenne")
            ax.set_title(label)
            ax.legend()
        fig.suptitle("ACF moyenne par groupe — niveaux", y=1.02, fontsize=11)
        fig.tight_layout()
        _save(fig, "course_acf_by_group")


def main() -> None:
    barplot_categories_by_group()
    acf_by_group()
    print("course group_compare: done")


if __name__ == "__main__":
    main()
