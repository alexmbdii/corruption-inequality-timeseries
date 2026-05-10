"""Render the robustness-matrix CSV as a heatmap of bootstrap p-values."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.paths import FIGURES, TABLES


def main() -> None:
    df = pd.read_csv(TABLES / "robustness_matrix.csv")
    # Pretty spec names
    df["spec_short"] = df["spec"].str.replace(r"^R\d+_", "", regex=True)
    pivot = df.pivot_table(
        index=["spec_short", "lags"],
        columns="direction",
        values="p_boot",
        aggfunc="first",
    )
    pivot = pivot[["corr_to_gini", "gini_to_corr"]]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(pivot) + 1.5))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
        vmin=0, vmax=0.20, cbar_kws={"label": "bootstrap p-value (clipped at 0.20)"},
        linewidths=0.4, linecolor="white",
        ax=ax,
    )
    ax.set_title("Phase 6 robustness — DH bootstrap p-values\n"
                 "(rows: spec × lags; greener = stronger reject of H₀)")
    ax.set_xlabel("direction tested"); ax.set_ylabel("spec")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06_robustness_heatmap.png", dpi=300)
    fig.savefig(FIGURES / "fig06_robustness_heatmap.pdf")
    plt.close(fig)
    print("wrote fig06_robustness_heatmap")


if __name__ == "__main__":
    main()
