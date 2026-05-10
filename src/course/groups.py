"""Three-way country grouping used by the course-aligned deliverable.

Brief asks for OECD / Émergents / Afrique. We define:
  - OECD: the 38 OECD member ISO3s present in the panel.
  - Afrique: every country whose WB region is SSF (Sub-Saharan Africa)
    plus the North-African MEA countries (Egypt, Morocco, Tunisia,
    Algeria, Libya), minus any country already in OECD.
  - Émergents: every other panel country.
"""
from __future__ import annotations

import pandas as pd

OECD_ISO3 = frozenset({
    "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE", "DNK", "EST",
    "FIN", "FRA", "DEU", "GRC", "HUN", "ISL", "IRL", "ISR", "ITA", "JPN",
    "KOR", "LVA", "LTU", "LUX", "MEX", "NLD", "NZL", "NOR", "POL", "PRT",
    "SVK", "SVN", "ESP", "SWE", "CHE", "TUR", "GBR", "USA",
})

NORTH_AFRICA_ISO3 = frozenset({"EGY", "MAR", "TUN", "DZA", "LBY"})

GROUP_ORDER = ["OECD", "Émergents", "Afrique"]

# Re-export the document-aligned palette so callers don't need to know which
# module owns the colors. See src/course/style.py for rationale.
from src.course.style import GROUP_COLORS_DOC as GROUP_COLORS  # noqa: E402


def assign_group(iso3: str, region: str | None) -> str:
    if iso3 in OECD_ISO3:
        return "OECD"
    if region == "SSF" or iso3 in NORTH_AFRICA_ISO3:
        return "Afrique"
    return "Émergents"


def add_group(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a `group` column added (OECD / Émergents / Afrique)."""
    out = df.copy()
    out["group"] = [assign_group(i, r) for i, r in zip(out["iso3"], out["region"])]
    return out
