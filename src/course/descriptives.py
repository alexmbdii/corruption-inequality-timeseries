"""Course deliverable, section 1: descriptive statistics & simple visualisations.

Outputs (all under outputs/{tables,figures}/, prefixed `course_`):
  course_summary_by_group.csv  — mean/std/min/max of v2x_corr, gini_disp by group
  course_pie_groups.png        — pie chart of country distribution by group
  course_timecurves.png        — mean time-series of corr and Gini per group, 1996–2022
  course_scatter_corr_gini.png — scatter v2x_corr vs gini_disp colored by region
  course_lagplots.png          — lag-1 plots on 6 representative countries
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.course.groups import GROUP_COLORS, GROUP_ORDER, add_group
from src.course.style import (FULL_PAGE_TALL, INK, SQUARE, TWO_PANEL, WIDE,
                              add_caption, apply_style)
from src.utils.paths import PROCESSED

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

DESCRIPTIVE_WINDOW = (1996, 2022)
REPRESENTATIVE = ["USA", "FRA", "BRA", "CHN", "NGA", "ZAF"]


def _load_panel() -> pd.DataFrame:
    p = pd.read_parquet(PROCESSED / "panel.parquet")
    return add_group(p)


def _save(fig, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png")
    fig.savefig(FIGURES / f"{stem}.pdf")
    plt.close(fig)


def summary_by_group(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in ["v2x_corr", "gini_disp"]:
        for g in GROUP_ORDER:
            sub = panel.loc[panel.group == g, var].dropna()
            rows.append({
                "variable": var, "group": g, "n_obs": int(sub.size),
                "n_countries": panel.loc[panel.group == g, "iso3"].nunique(),
                "mean": sub.mean(), "std": sub.std(),
                "min": sub.min(), "max": sub.max(),
            })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "course_summary_by_group.csv", index=False)
    return out


def pie_groups(panel: pd.DataFrame) -> None:
    counts = (panel.drop_duplicates("iso3").groupby("group").iso3.count()
              .reindex(GROUP_ORDER))
    with apply_style():
        fig, ax = plt.subplots(figsize=SQUARE)
        wedges, _, autotexts = ax.pie(
            counts,
            labels=[f"{g}\n($n={c}$)" for g, c in counts.items()],
            colors=[GROUP_COLORS[g] for g in counts.index],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},
            textprops={"fontsize": 10, "color": INK},
        )
        for t in autotexts:
            t.set_color("white")
            t.set_fontweight("bold")
        ax.set_title(f"Répartition des pays par groupe ($n={int(counts.sum())}$)",
                     loc="center", pad=12)
        fig.tight_layout()
        _save(fig, "course_pie_groups")


def time_curves(panel: pd.DataFrame) -> None:
    y0, y1 = DESCRIPTIVE_WINDOW
    sub = panel[(panel.year >= y0) & (panel.year <= y1)].copy()
    grp_size = (sub.drop_duplicates("iso3").groupby("group").iso3.count())
    cov = (sub.groupby(["group", "year"]).gini_disp.apply(lambda s: s.notna().sum())
            .reset_index(name="n_gini"))
    cov["frac"] = cov.apply(lambda r: r.n_gini / grp_size[r.group], axis=1)
    bad_keys = set(map(tuple, cov[cov.frac < 0.5][["group", "year"]].values))
    sub = sub[~sub.apply(lambda r: (r.group, r.year) in bad_keys, axis=1)].copy()
    means = sub.groupby(["group", "year"])[["v2x_corr", "gini_disp"]].mean().reset_index()

    with apply_style():
        fig, axes = plt.subplots(1, 2, figsize=TWO_PANEL)
        for var, ax, label in [("v2x_corr", axes[0], r"Corruption (V-Dem $v2x_{\mathrm{corr}}$)"),
                                ("gini_disp", axes[1], r"Inégalités (Gini disponible)")]:
            for g in GROUP_ORDER:
                d = means[means.group == g]
                ax.plot(d.year, d[var], label=g, color=GROUP_COLORS[g], lw=1.6)
            ax.set_title(label)
            ax.set_xlabel("Année")
            ax.legend(loc="best")
        axes[0].set_ylabel(r"$v2x_{\mathrm{corr}}$ (moyenne)")
        axes[1].set_ylabel("Gini (moyenne)")
        fig.tight_layout()
        _save(fig, "course_timecurves")


def scatter_corr_gini(panel: pd.DataFrame) -> None:
    y0, y1 = DESCRIPTIVE_WINDOW
    sub = panel[(panel.year >= y0) & (panel.year <= y1)].dropna(subset=["v2x_corr", "gini_disp"])
    with apply_style():
        fig, ax = plt.subplots(figsize=(6.3, 4.4))
        for g in GROUP_ORDER:
            d = sub[sub.group == g]
            ax.scatter(d.v2x_corr, d.gini_disp, alpha=0.30, s=10,
                       color=GROUP_COLORS[g],
                       label=f"{g} ($n_{{pays}}={d.iso3.nunique()}$)",
                       edgecolors="none")
        x = sub.v2x_corr.values
        y = sub.gini_disp.values
        a, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, a * xs + b, color=INK, lw=1.2, ls="--",
                label=fr"OLS pool: $\hat{{\beta}} = {a:.1f}$")
        ax.set_xlabel(r"Corruption ($v2x_{\mathrm{corr}}$, $\uparrow$ = plus corrompu)")
        ax.set_ylabel(r"Gini disponible ($gini_{\mathrm{disp}}$)")
        ax.set_title(f"Corruption × inégalités, {y0}–{y1}")
        ax.legend(loc="upper left")
        fig.tight_layout()
        _save(fig, "course_scatter_corr_gini")


def lag_plots(panel: pd.DataFrame) -> None:
    from matplotlib.ticker import MaxNLocator
    with apply_style():
        fig, axes = plt.subplots(2, len(REPRESENTATIVE),
                                  figsize=(6.3, 4.6), sharex=False, sharey=False)
        for j, iso in enumerate(REPRESENTATIVE):
            sub = panel[panel.iso3 == iso].sort_values("year")
            grp_color = GROUP_COLORS[panel.loc[panel.iso3 == iso, "group"].iloc[0]]
            for i, (var, label) in enumerate([("v2x_corr", "C"),
                                                ("gini_disp", "I")]):
                ax = axes[i, j]
                s = sub[var].dropna().values.astype(float)
                if s.size < 3:
                    ax.set_visible(False)
                    continue
                ax.scatter(s[:-1], s[1:], alpha=0.75, s=12, color=grp_color,
                           edgecolors="none")
                lo, hi = float(s.min()), float(s.max())
                pad = 0.05 * (hi - lo if hi > lo else 1)
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                        color=INK, lw=0.6, ls="--", alpha=0.5)
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.tick_params(labelsize=7)
                if i == 1:
                    ax.set_xlabel(fr"${label}_{{t}}$", fontsize=9)
                if j == 0:
                    ax.set_ylabel(fr"${label}_{{t+1}}$", fontsize=9)
                if i == 0:
                    ax.set_title(iso, fontsize=10, loc="center", pad=4)
        fig.suptitle(r"Lag-plots ($x_{t}$ vs $x_{t+1}$) — corruption (haut), inégalités (bas)",
                     fontsize=10.5, y=1.00, x=0.5, ha="center")
        fig.tight_layout()
        _save(fig, "course_lagplots")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    panel = _load_panel()
    summary_by_group(panel)
    pie_groups(panel)
    time_curves(panel)
    scatter_corr_gini(panel)
    lag_plots(panel)
    print("course descriptives: done")


if __name__ == "__main__":
    main()
