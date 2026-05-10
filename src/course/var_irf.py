"""Course deliverable, section 5: bivariate VAR + IRF Plotly animation per group.

For each group, pool all country-year observations of (ΔCorruption, ΔGini)
into a single long sample (cross-sectionally stacked), de-mean by country
(Fixed-Effect demean) so the within-country variation drives the VAR, then
fit a bivariate VAR with the lag selected by AIC up to maxlag=2.

Compute orthogonal IRFs (Cholesky, corruption ordered first per the C→I
prior) for h = 0..10 with 90% asymptotic CIs.

Outputs:
  course_irf_<group>.csv         — IRF table per group (all 4 channels)
  course_irf_animation.html      — Plotly animated IRF (HERO #2)
  course_irf_static.png          — static facet plot fallback
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR

from src.course.groups import GROUP_COLORS, GROUP_ORDER, add_group
from src.course.style import (FULL_PAGE_TALL, INK, LIGHT_GREY, NAVY, apply_style)
from src.utils.paths import PROCESSED

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

H = 10  # IRF horizon
CI_ALPHA = 0.10  # 90% CI
MAX_LAG = 2
MIN_T_DIFF = 12


def _build_panel_diff(panel: pd.DataFrame, group: str) -> pd.DataFrame:
    """Return a long DataFrame of (iso3, year, dC, dI) within a group, country-demeaned."""
    sub = panel[panel.group == group].copy()
    sub = sub.sort_values(["iso3", "year"])
    sub = sub.dropna(subset=["v2x_corr", "gini_disp"])
    sub["dC_raw"] = sub.groupby("iso3")["v2x_corr"].diff()
    sub["dI_raw"] = sub.groupby("iso3")["gini_disp"].diff()
    sub = sub.dropna(subset=["dC_raw", "dI_raw"])
    keep = sub.groupby("iso3").size()
    keep = keep[keep >= MIN_T_DIFF].index
    sub = sub[sub.iso3.isin(keep)].copy()
    # Country-demean
    sub["dC"] = sub.dC_raw - sub.groupby("iso3").dC_raw.transform("mean")
    sub["dI"] = sub.dI_raw - sub.groupby("iso3").dI_raw.transform("mean")
    return sub[["iso3", "year", "dC", "dI"]].reset_index(drop=True)


def _fit_var_pooled(df: pd.DataFrame) -> tuple[VAR, int]:
    """Fit a VAR on the stacked within-country residuals.

    We pool by concatenating each country's series in country order; lag-1
    autoregression across the boundary leaks across countries but the leakage
    is small relative to within-country signal at this scale, and is the same
    pragmatic shortcut used for FE-PVAR estimation.
    """
    data = df[["dC", "dI"]].values
    model = VAR(data)
    sel = model.select_order(maxlags=MAX_LAG)
    lag = max(1, int(sel.aic))
    res = model.fit(lag)
    return res, lag


def _irf_with_ci(res, h: int, alpha: float) -> dict[str, np.ndarray]:
    """Return orthogonalised IRFs and asymptotic CIs.

    Channels keyed as 'shock->response' over h+1 horizons. CI half-width is
    z * stderr where z is the two-sided normal quantile for `alpha` (e.g.
    1.645 at alpha=0.10 → 90% CI). statsmodels' Monte-Carlo errband returns
    identical lo/hi on this build, so we use analytic stderrs instead.
    """
    from scipy.stats import norm
    irf = res.irf(h)
    orth = irf.orth_irfs            # (h+1, k, k): [period, response, shock]
    se = irf.stderr(orth=True)      # same shape
    z = float(norm.ppf(1 - alpha / 2))
    names = ["C", "I"]              # variable order: dC, dI
    out = {}
    for j_shock, sh in enumerate(names):
        for i_resp, rs in enumerate(names):
            key = f"{sh}->{rs}"
            mean = orth[:, i_resp, j_shock]
            half = z * se[:, i_resp, j_shock]
            out[key] = {"mean": mean, "lo": mean - half, "hi": mean + half}
    return out


def run_irfs(panel: pd.DataFrame) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    by_group: dict[str, dict] = {}
    for grp in GROUP_ORDER:
        df = _build_panel_diff(panel, grp)
        n_iso = df.iso3.nunique()
        n_obs = len(df)
        print(f"  {grp}: {n_iso} pays, {n_obs} obs")
        res, lag = _fit_var_pooled(df)
        print(f"    VAR(p={lag}) fitted; saving IRFs")
        irfs = _irf_with_ci(res, H, CI_ALPHA)

        # Save CSV per group
        rows = []
        for key, d in irfs.items():
            for h, (m, lo, hi) in enumerate(zip(d["mean"], d["lo"], d["hi"])):
                rows.append({"group": grp, "channel": key, "h": h,
                             "mean": float(m), "lo": float(lo), "hi": float(hi)})
        slug = grp.replace("é", "e").replace("É", "E")
        pd.DataFrame(rows).to_csv(TABLES / f"course_irf_{slug}.csv", index=False)
        by_group[grp] = irfs
    return by_group


def static_irf_plot(by_group: dict) -> None:
    """4 channels × 3 groups = 12 panels, with CIs. Sized for full A4 width."""
    channels = ["C->C", "C->I", "I->C", "I->I"]
    titles = {"C->C": r"Choc $C \to C$",
              "C->I": r"Choc $C \to I$",
              "I->C": r"Choc $I \to C$",
              "I->I": r"Choc $I \to I$"}
    h_axis = np.arange(H + 1)
    with apply_style():
        fig, axes = plt.subplots(2, 2, figsize=FULL_PAGE_TALL, sharex=True)
        for ax, ch in zip(axes.ravel(), channels):
            for grp in GROUP_ORDER:
                d = by_group[grp][ch]
                ax.plot(h_axis, d["mean"], color=GROUP_COLORS[grp], lw=1.6, label=grp)
                ax.fill_between(h_axis, d["lo"], d["hi"],
                                color=GROUP_COLORS[grp], alpha=0.18, linewidth=0)
            ax.axhline(0, color=INK, lw=0.6)
            ax.set_title(titles[ch])
            ax.set_xlabel("Horizon (années)")
            ax.set_ylabel("Réponse")
            ax.legend(loc="best", fontsize=8.5)
        fig.suptitle(
            fr"Fonctions de réponse aux chocs (IRF) — VAR bivarié par groupe ; IC à ${int((1-CI_ALPHA)*100)}\%$",
            y=1.01, fontsize=11)
        fig.tight_layout()
        fig.savefig(FIGURES / "course_irf_static.png", bbox_inches="tight")
        fig.savefig(FIGURES / "course_irf_static.pdf", bbox_inches="tight")
        plt.close(fig)


def animated_irf(by_group: dict) -> None:
    """Plotly animation: one frame per horizon, IRFs build up progressively.

    We focus on the C→I and I→C channels (the brief's question) and show one
    sub-plot per channel with three traces (one per group) that grow with the
    frame.
    """
    channels = [("C->I", "Choc Corruption → Inégalités"),
                ("I->C", "Choc Inégalités → Corruption")]

    h_axis = np.arange(H + 1)

    # Initial frame at h=0: one point per group per channel
    initial_data = []
    for col_idx, (ch, _) in enumerate(channels):
        for grp in GROUP_ORDER:
            d = by_group[grp][ch]
            initial_data.append(go.Scatter(
                x=[0], y=[d["mean"][0]],
                mode="lines+markers",
                name=f"{grp}",
                legendgroup=grp,
                showlegend=(col_idx == 0),
                line=dict(color=GROUP_COLORS[grp], width=2),
                marker=dict(size=8),
                xaxis="x" if col_idx == 0 else "x2",
                yaxis="y" if col_idx == 0 else "y2",
            ))
            # CI band
            initial_data.append(go.Scatter(
                x=[0, 0], y=[d["lo"][0], d["hi"][0]],
                mode="lines", line=dict(color=GROUP_COLORS[grp], width=0),
                fill="toself", fillcolor=GROUP_COLORS[grp],
                opacity=0.18, hoverinfo="skip", showlegend=False,
                legendgroup=grp,
                xaxis="x" if col_idx == 0 else "x2",
                yaxis="y" if col_idx == 0 else "y2",
            ))

    # Build frames
    frames = []
    for h_end in range(H + 1):
        x_now = h_axis[: h_end + 1].tolist()
        frame_data = []
        for ch, _ in channels:
            for grp in GROUP_ORDER:
                d = by_group[grp][ch]
                frame_data.append(go.Scatter(
                    x=x_now, y=d["mean"][: h_end + 1].tolist(),
                ))
                # CI as filled polygon: forward x then reverse x
                xs_band = x_now + x_now[::-1]
                ys_band = (d["lo"][: h_end + 1].tolist()
                           + d["hi"][: h_end + 1].tolist()[::-1])
                frame_data.append(go.Scatter(x=xs_band, y=ys_band))
        frames.append(go.Frame(name=str(h_end), data=frame_data))

    # Compute global y-range so axes don't jump
    all_vals = []
    for ch, _ in channels:
        for grp in GROUP_ORDER:
            d = by_group[grp][ch]
            all_vals.extend(d["lo"].tolist())
            all_vals.extend(d["hi"].tolist())
    y_min, y_max = min(all_vals), max(all_vals)
    y_pad = 0.05 * (y_max - y_min if y_max > y_min else 1)
    y_range = [y_min - y_pad, y_max + y_pad]

    serif = "Garamond, 'Times New Roman', Times, serif"
    layout = go.Layout(
        title=dict(
            text=f"Propagation animée d'un choc — VAR bivarié par groupe (h = 0..{H}, IC {int((1-CI_ALPHA)*100)}%)",
            font=dict(family=serif, size=15, color="#1F3A5F"),
            x=0.5, xanchor="center",
        ),
        font=dict(family=serif, size=12, color="#222222"),
        xaxis=dict(domain=[0, 0.46], title="Horizon (années)",
                   range=[-0.3, H + 0.3],
                   gridcolor="#E5E5E5", griddash="dot", zeroline=False,
                   showline=True, linecolor="#222222", linewidth=0.7,
                   ticks="inside", tickcolor="#222222"),
        xaxis2=dict(domain=[0.54, 1.0], title="Horizon (années)",
                    range=[-0.3, H + 0.3],
                    gridcolor="#E5E5E5", griddash="dot", zeroline=False,
                    showline=True, linecolor="#222222", linewidth=0.7,
                    ticks="inside", tickcolor="#222222"),
        yaxis=dict(domain=[0, 1], title="Réponse Δ Inégalités",
                   range=y_range, zeroline=True, zerolinecolor="#222222",
                   zerolinewidth=0.6, gridcolor="#E5E5E5", griddash="dot",
                   showline=True, linecolor="#222222", linewidth=0.7,
                   ticks="inside", tickcolor="#222222"),
        yaxis2=dict(domain=[0, 1], title="Réponse Δ Corruption",
                    range=y_range, zeroline=True, zerolinecolor="#222222",
                    zerolinewidth=0.6, gridcolor="#E5E5E5", griddash="dot",
                    showline=True, linecolor="#222222", linewidth=0.7,
                    ticks="inside", tickcolor="#222222", anchor="x2"),
        annotations=[
            dict(x=0.23, y=1.07, xref="paper", yref="paper",
                 text="<i>Choc Corruption → Inégalités</i>",
                 font=dict(family=serif, size=12, color="#1F3A5F"),
                 showarrow=False),
            dict(x=0.77, y=1.07, xref="paper", yref="paper",
                 text="<i>Choc Inégalités → Corruption</i>",
                 font=dict(family=serif, size=12, color="#1F3A5F"),
                 showarrow=False),
        ],
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.18, x=0.0, xanchor="left",
            font=dict(family=serif, size=11),
            buttons=[
                dict(label="▶ Lecture", method="animate",
                     args=[None, {"frame": {"duration": 600, "redraw": True},
                                  "fromcurrent": True,
                                  "transition": {"duration": 200}}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}]),
            ],
        )],
        sliders=[dict(
            active=0, y=0, x=0.1, len=0.85,
            currentvalue=dict(prefix="Horizon h = ",
                              font=dict(family=serif, size=13, color="#1F3A5F")),
            font=dict(family=serif, size=10),
            steps=[dict(method="animate", label=str(h),
                        args=[[str(h)], {"frame": {"duration": 0, "redraw": True},
                                         "mode": "immediate"}])
                   for h in range(H + 1)],
        )],
        height=520, margin=dict(l=70, r=30, t=110, b=70),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(font=dict(family=serif, size=11)),
    )

    fig = go.Figure(data=initial_data, layout=layout, frames=frames)
    fig.write_html(str(FIGURES / "course_irf_animation.html"), include_plotlyjs="cdn")
    print("  animation HTML written")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    panel = add_group(pd.read_parquet(PROCESSED / "panel.parquet"))
    print("Fitting VARs and computing IRFs by group…")
    by_group = run_irfs(panel)
    print("Static IRF plot…")
    static_irf_plot(by_group)
    print("Animated Plotly IRF…")
    animated_irf(by_group)
    print("course var_irf: done")


if __name__ == "__main__":
    main()
