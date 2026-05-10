"""Course deliverable, section 3: country-by-country Granger causality.

For each country, on the joint window where both v2x_corr and gini_disp are
observed, take first differences and run statsmodels' grangercausalitytests
in both directions (lags 1 and 2; report the smallest p-value across the
two — see _granger_p docstring for the multiple-testing caveat).

Classify each country into one of:
    "C→I"     — corruption Granger-causes inequality only
    "I→C"     — inequality Granger-causes corruption only
    "Bidir"   — both
    "Aucune"  — neither

Outputs:
  course_granger_country.csv     — per-country p-values + classification
  course_granger_summary.csv     — counts per category × group
  course_pie_categories.png      — pie chart of category proportions
  course_choropleth.html         — Plotly interactive world map (HERO #1)
  course_choropleth.png          — static fallback
  course_pvalue_heatmap.png      — heatmap of p-values per country × direction
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from statsmodels.tsa.stattools import grangercausalitytests

from src.course.groups import GROUP_COLORS, GROUP_ORDER, add_group
from src.course.style import (CAT_COLORS_DOC, INK, SQUARE, TALL, apply_style)
from src.utils.paths import PROCESSED

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

ALPHA = 0.05
MAX_LAG = 2
MIN_T_DIFF = 12  # minimum differenced obs per country
CAT_ORDER = ["C→I", "I→C", "Bidir", "Aucune"]
CAT_COLORS = CAT_COLORS_DOC


def _joint_diff(panel: pd.DataFrame, iso: str) -> pd.DataFrame:
    sub = (panel[panel.iso3 == iso].sort_values("year")
           [["year", "v2x_corr", "gini_disp"]].dropna()).reset_index(drop=True)
    if len(sub) < MIN_T_DIFF + 1:
        return pd.DataFrame()
    sub["dC"] = sub.v2x_corr.diff()
    sub["dI"] = sub.gini_disp.diff()
    return sub.dropna().reset_index(drop=True)


def _granger_p(data: np.ndarray, max_lag: int) -> tuple[float, int] | tuple[None, None]:
    """Run gctest on pair [y, x]; null = x does not Granger-cause y.

    Tries lags 1..max_lag, returns the smallest p-value across lags and the lag
    that produced it. statsmodels' grangercausalitytests requires len(data) >
    3 * lag, so we cap accordingly.
    """
    n = len(data)
    max_feasible = min(max_lag, max(1, n // 3 - 1))
    if max_feasible < 1:
        return None, None
    try:
        res = grangercausalitytests(data, maxlag=max_feasible, verbose=False)
    except Exception:
        return None, None
    best_p, best_lag = 1.0, None
    for k, r in res.items():
        try:
            p = float(r[0]["ssr_ftest"][1])
        except Exception:
            continue
        if p < best_p:
            best_p, best_lag = p, int(k)
    return (best_p, best_lag) if best_lag is not None else (None, None)


def run_country_granger(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iso, sub in panel.groupby("iso3"):
        group = sub.group.iloc[0]
        cname = sub.country_name.iloc[0] if "country_name" in sub else iso
        d = _joint_diff(panel, iso)
        if d.empty:
            rows.append({"iso3": iso, "country": cname, "group": group,
                         "T_diff": 0, "p_C_to_I": None, "lag_C_to_I": None,
                         "p_I_to_C": None, "lag_I_to_C": None,
                         "category": "n/a"})
            continue
        # C → I: x = dC, y = dI; statsmodels expects [y, x]
        p_ci, lag_ci = _granger_p(d[["dI", "dC"]].values, MAX_LAG)
        # I → C: x = dI, y = dC
        p_ic, lag_ic = _granger_p(d[["dC", "dI"]].values, MAX_LAG)
        if p_ci is None or p_ic is None:
            cat = "n/a"
        else:
            ci = p_ci < ALPHA
            ic = p_ic < ALPHA
            cat = ("Bidir" if ci and ic else
                   "C→I" if ci and not ic else
                   "I→C" if ic and not ci else "Aucune")
        rows.append({"iso3": iso, "country": cname, "group": group,
                     "T_diff": len(d),
                     "p_C_to_I": p_ci, "lag_C_to_I": lag_ci,
                     "p_I_to_C": p_ic, "lag_I_to_C": lag_ic,
                     "category": cat})
    df = pd.DataFrame(rows).sort_values("iso3").reset_index(drop=True)
    df.to_csv(TABLES / "course_granger_country.csv", index=False)
    return df


def summary_table(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grp in GROUP_ORDER + ["TOTAL"]:
        sub = g if grp == "TOTAL" else g[g.group == grp]
        counts = sub.category.value_counts().to_dict()
        row = {"group": grp, "n_countries": len(sub)}
        for c in CAT_ORDER + ["n/a"]:
            row[c] = counts.get(c, 0)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "course_granger_summary.csv", index=False)
    return out


def pie_categories(g: pd.DataFrame) -> None:
    sub = g[g.category.isin(CAT_ORDER)]
    counts = sub.category.value_counts().reindex(CAT_ORDER).fillna(0)
    with apply_style():
        fig, ax = plt.subplots(figsize=SQUARE)
        labels = [f"{c}\n($n={int(n)}$)" for c, n in counts.items()]
        wedges, _, autotexts = ax.pie(
            counts, labels=labels,
            colors=[CAT_COLORS[c] for c in counts.index],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},
            textprops={"fontsize": 9.5, "color": INK},
        )
        for t in autotexts:
            t.set_color("white")
            t.set_fontweight("bold")
            t.set_fontsize(9.5)
        ax.set_title(f"Direction de causalité Granger par pays ($n={int(counts.sum())}$)",
                     pad=12)
        fig.tight_layout()
        fig.savefig(FIGURES / "course_pie_categories.png")
        fig.savefig(FIGURES / "course_pie_categories.pdf")
        plt.close(fig)


def choropleth(g: pd.DataFrame) -> None:
    df = g[g.category.isin(CAT_ORDER + ["n/a"])].copy()
    df["category"] = pd.Categorical(df.category, categories=CAT_ORDER + ["n/a"], ordered=True)
    fig = px.choropleth(
        df, locations="iso3", color="category",
        hover_name="country",
        hover_data={"iso3": True, "group": True,
                    "p_C_to_I": ":.3f", "p_I_to_C": ":.3f",
                    "T_diff": True, "category": False},
        category_orders={"category": CAT_ORDER + ["n/a"]},
        color_discrete_map={**CAT_COLORS, "n/a": "#EAEAEA"},
        title=f"Causalité de Granger Corruption × Inégalités par pays (α = {ALPHA})",
    )
    fig.update_geos(
        showcountries=True, countrycolor="#FFFFFF", countrywidth=0.6,
        showcoastlines=True, coastlinecolor="#888888", coastlinewidth=0.4,
        showland=True, landcolor="#F8F8F8",
        showocean=True, oceancolor="#FFFFFF",
        projection_type="natural earth",
        showframe=False,
    )
    fig.update_layout(
        legend_title_text="Catégorie causale",
        font=dict(family="Garamond, Times New Roman, serif", size=13, color="#1F3A5F"),
        title=dict(font=dict(size=15, color="#1F3A5F"), x=0.5, xanchor="center"),
        margin=dict(l=0, r=0, t=60, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.write_html(str(FIGURES / "course_choropleth.html"), include_plotlyjs="cdn")
    try:
        fig.write_image(str(FIGURES / "course_choropleth.png"),
                        width=1400, height=800, scale=2)
    except Exception as e:
        print(f"(static png export skipped: {e})")


def pvalue_heatmap(g: pd.DataFrame) -> None:
    """Wide-layout heatmap that fits A4 portrait: split countries into 3 columns
    (one per group), each column has 2 sub-cols (C→I, I→C). Avoids the 1×128
    skinny strip that doesn't fit the report well."""
    df = g.dropna(subset=["p_C_to_I", "p_I_to_C"]).copy()
    cols = []
    for grp in GROUP_ORDER:
        sub = df[df.group == grp].sort_values("iso3").reset_index(drop=True)
        cols.append((grp, sub))
    max_n = max(len(s) for _, s in cols)

    with apply_style():
        fig, axes = plt.subplots(1, 3, figsize=(6.3, 0.10 * max_n + 1.6),
                                   gridspec_kw={"wspace": 0.55})
        for ax, (grp, sub) in zip(axes, cols):
            mat = sub[["p_C_to_I", "p_I_to_C"]].values
            # Pad to max_n with NaN so columns share height
            pad = max_n - len(sub)
            if pad > 0:
                mat = np.vstack([mat, np.full((pad, 2), np.nan)])
            im = ax.imshow(mat, aspect="auto", cmap="RdYlBu_r",
                           vmin=0, vmax=0.20, interpolation="none")
            ax.set_xticks([0, 1])
            ax.set_xticklabels([r"C$\to$I", r"I$\to$C"], fontsize=8)
            ax.set_yticks(range(max_n))
            labels = sub.iso3.tolist() + [""] * pad
            ax.set_yticklabels(labels, fontsize=5)
            ax.set_title(f"{grp} ($n={len(sub)}$)", fontsize=9.5,
                         loc="center", pad=4)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        cbar = fig.colorbar(im, ax=axes, shrink=0.65, pad=0.02, aspect=30)
        cbar.set_label(fr"$p$-value Granger (tronquée à 0.20 ; $\alpha = {ALPHA}$)",
                        fontsize=8.5)
        cbar.ax.tick_params(labelsize=7)
        fig.suptitle("Heatmap des $p$-values Granger — pays × direction × groupe",
                     fontsize=10.5, y=1.02)
        fig.savefig(FIGURES / "course_pvalue_heatmap.png", bbox_inches="tight")
        fig.savefig(FIGURES / "course_pvalue_heatmap.pdf", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    panel = add_group(pd.read_parquet(PROCESSED / "panel.parquet"))
    print("Country-by-country Granger…")
    g = run_country_granger(panel)
    print("Summary table…")
    s = summary_table(g)
    print(s.to_string(index=False))
    print("Pie chart…")
    pie_categories(g)
    print("Choropleth…")
    choropleth(g)
    print("Heatmap…")
    pvalue_heatmap(g)
    print("course granger_country: done")


if __name__ == "__main__":
    main()
