# Deliverables — guide pour les collègues

Ce document est le point d'entrée pour la **version cours** du projet,
ajoutée le 2026-05-09 par-dessus la version méthodes-paper initiale.

## Qu'est-ce qui est nouveau ?

Le brief du cours demande une analyse construite avec les outils du cours
(ACF/PACF, ADF/KPSS, Granger, VAR/IRF), structurée en 5 sections, avec
deux figures « hero » : une choroplèthe interactive et une animation IRF.
La version initiale du repo (REPORT.md) utilise des méthodes plus avancées
(Dumitrescu–Hurlin, Toda–Yamamoto, Pedroni, projections locales,
synthetic-control) qui sont hors-programme.

**Les deux versions coexistent.** Rien n'a été supprimé. Le contenu cours
est préfixé `course_` partout (figures, tables, code), donc identifiable
en un coup d'œil.

## Où regarder

| Si tu veux… | Ouvre |
|---|---|
| Lire le rapport cours (5 sections) | `REPORT_COURSE.md` |
| Voir la choroplèthe interactive | `outputs/figures/course_choropleth.html` |
| Voir l'animation IRF | `outputs/figures/course_irf_animation.html` |
| Toutes les tables cours | `outputs/tables/course_*.csv` |
| Toutes les figures cours | `outputs/figures/course_*.{png,pdf}` |
| Le code | `src/course/` |
| Le rapport méthodes-paper original | `REPORT.md` |
| L'oral original | `ORAL_DEFENSE_PREP.md` |

## Comment regénérer

Les cinq sections du rapport cours correspondent à cinq scripts dans
`src/course/`. Ils s'exécutent indépendamment et écrivent dans
`outputs/{tables,figures}/`.

```bash
# Setup (déjà fait, mais si besoin)
uv sync

# Exécution dans l'ordre — chaque script est ~5–30s sauf Granger ~1min
.venv/bin/python -m src.course.descriptives        # Section 1
.venv/bin/python -m src.course.series_properties   # Section 2
.venv/bin/python -m src.course.granger_country     # Section 3 + heros
.venv/bin/python -m src.course.group_compare       # Section 4
.venv/bin/python -m src.course.var_irf             # Section 5 + animation
```

Les notebooks `notebooks/01..06_*.ipynb` restent les notebooks de la version
méthodes-paper. Les scripts cours sont uniquement en `src/course/` (pas
de notebooks dédiés — le code est court et se lit directement).

## Structure ajoutée

```
src/course/
├── groups.py            # OECD / Émergents / Afrique
├── descriptives.py      # Section 1 — stats, pie, courbes, scatter, lag-plots
├── series_properties.py # Section 2 — ACF/PACF, ADF/KPSS, histogramme
├── granger_country.py   # Section 3 — Granger pays-par-pays + choropleth
├── group_compare.py     # Section 4 — barplot + ACF par groupe
└── var_irf.py           # Section 5 — VAR bivarié + animation Plotly

outputs/figures/course_*.{png,pdf,html}
outputs/tables/course_*.csv

REPORT_COURSE.md         # Le rapport cours
DELIVERABLES.md          # Ce fichier
pyproject.toml           # +plotly +kaleido
```

## Décisions méthodologiques (rapide)

- **Trois groupes** (`src/course/groups.py`) : OECD = 38 pays membres
  présents dans le panel ; Afrique = SSF + Maghreb (EGY, MAR, TUN, DZA,
  LBY) ; Émergents = le reste. Total 38 + 56 + 34 = 128.
- **Fenêtre** : tests pays-par-pays sur la fenêtre maximale de chaque pays
  (1990–2023 où dispo) ; descriptives sur 1996–2022 (couverture SWIID
  plus propre).
- **Granger** : sur séries différenciées d'ordre 1 (validé par le tableau
  ADF/KPSS cours). Lags 1 et 2 testés ; on retient la plus petite p-value
  (min-p test, α effectif ≈ 9.75% pour un seuil brut à 5%). Décision α = 5%.
- **VAR par groupe** : pool des observations différenciées du groupe,
  démean par pays (effet fixe), VAR avec ordre choisi par AIC ≤ 2,
  Cholesky avec corruption ordonnée en premier. IRF avec IC 90%
  asymptotiques (z·SE).

## Points d'attention pour la défense orale

- La choroplèthe et l'animation sont des HTML : il faut les ouvrir dans un
  navigateur, pas dans VS Code preview (sinon Plotly ne charge pas).
- Le résultat « Émergents : C→I significatif » est la nouvelle conclusion
  forte du rapport cours. Elle est cohérente avec le résultat post-Soviet
  sub-sample du REPORT.md (les post-Soviets sont dans Émergents).
- 80% des pays ont 0 causalité détectable : ne pas survendre la lecture
  pays-par-pays. Le bon framing est « hétérogénéité » et non « consensus
  causal mondial ».
