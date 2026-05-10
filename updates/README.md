Version courte de ce qui a été ajouté pour que vous puissiez le brancher dans le rapport
du Time Series Project.

## Résumé

Le repo contient maintenant une seconde passe complète de l'analyse en n'utilisant que
**les outils vus en cours** (ACF/PACF, ADF, KPSS, Granger bivarié, VAR/IRF), avec
classification des pays en **OCDE / Émergents / Afrique**. Deux figures « hero »
interactives ont été construites. Tout est restylé pour matcher l'esthétique
serif/navy du rapport, donc vous pouvez glisser-déposer les figures directement dans
le doc Word/Google.

- **Le code** vit dans `src/course/` (5 modules courts, un par section du rapport)
- **Les sorties brutes** sont dans `outputs/figures/course_*.{png,pdf,html}` et
  `outputs/tables/course_*.csv`
- **Ce qui est dans ce dossier** est le sous-ensemble curated — les figures les plus
  utiles et les tables qui sous-tendent les chiffres headline, toutes renommées par
  section du rapport

## Contenu du dossier

```
for_colleagues/
├── README.md                                      ← vous êtes ici
├── headline_findings.md                           ← chiffres prêts à coller pour §2–§5
├── figures/
│   ├── s2_pie_country_distribution.png            §2 — répartition des pays
│   ├── s2_timecurves_by_group.png                 §2 — moyennes corruption & Gini par groupe
│   ├── s2_scatter_corr_vs_gini.png                §2 — scatter coloré par groupe
│   ├── s3_acf_pacf_grid.png                       §3 — ACF/PACF, 6 pays repr.
│   ├── s3_adf_pvalue_histogram.png                §3 — distrib. p-values ADF
│   ├── s3_acf_before_after_diff.png               §3 — ACF avant/après différenciation
│   ├── s4_HERO_choropleth_interactive.html  ★     §4 — la HERO #1 (à montrer à l'oral)
│   ├── s4_HERO_choropleth_static.png              §4 — version statique pour le PDF
│   ├── s4_pie_granger_categories.png              §4 — pie des 4 catégories
│   ├── s4_barplot_categories_by_group.png         §4 — barplot par groupe
│   ├── s4_pvalue_heatmap_by_group.png             §4 — heatmap des p-values
│   ├── s4_acf_by_group.png                        §4 — ACF moyenne par groupe
│   ├── s5_HERO_irf_animation.html           ★     §5 — la HERO #2 (à montrer à l'oral)
│   └── s5_irf_static_panels.png                   §5 — IRF statiques pour le PDF
└── tables/
    ├── s2_summary_by_group.csv                    moyennes/écarts-types par groupe
    ├── s3_adf_kpss_per_country.csv                ADF + KPSS pour chaque pays
    ├── s3_adf_kpss_summary_by_group.csv           comptes stationnaire/non/ambigu par groupe
    ├── s4_granger_per_country.csv                 p-values Granger par pays + catégorie
    ├── s4_granger_summary_by_group.csv            comptes 4-catégories par groupe
    └── s5_irf_{OECD,Emergents,Afrique}.csv        valeurs IRF h=0..10 + IC 90 %
```

Chaque figure est fournie en **PNG (pour Word)** et en **PDF (pour LaTeX)**.

## Où placer chaque figure dans votre rapport

| Section | Figure(s) suggérée(s) | Pourquoi |
|---|---|---|
| §2 Données & variables | `s2_pie_country_distribution.png` + `s2_summary_by_group.csv` (transcrit en tableau) | Présenter les 3 groupes (OCDE/Émergents/Afrique) et les volumes |
| §2 Données & variables | `s2_timecurves_by_group.png` | Vue d'ensemble : niveaux de corruption & Gini par groupe sur 25 ans |
| §3 Pre-testing | `s3_acf_pacf_grid.png` | Illustre la persistance des deux séries |
| §3 Pre-testing | `s3_adf_pvalue_histogram.png` | Justifie le besoin de différencier (la majorité des pays sont non-stationnaires) |
| §3 First-differencing | `s3_acf_before_after_diff.png` | C'est la **Figure 2** déjà référencée dans la légende du rapport |
| §4 Granger causality | **`s4_HERO_choropleth_static.png`** ★ | Hero figure — direction causale par pays |
| §4 Granger causality | `s4_pie_granger_categories.png` + `s4_barplot_categories_by_group.png` | Vue d'ensemble : 80 % Aucune, équilibre C→I/I→C, hétérogénéité par groupe |
| §4 Granger causality | `s4_pvalue_heatmap_by_group.png` | Détails — pour qui veut voir les p-values pays par pays |
| §5 Conclusion | **`s5_irf_static_panels.png`** ★ | Le résultat headline : choc Corruption→Inégalités significatif chez les Émergents seulement |
| §5 Conclusion (oral) | `s5_HERO_irf_animation.html` | Animation Plotly — à projeter, pas à imprimer |
| Annexe (optionnel) | `s2_scatter_corr_vs_gini.png`, `s4_acf_by_group.png` | Compléments visuels |

## Comment ouvrir les fichiers HERO HTML

Les deux fichiers `*HERO*.html` sont des **figures Plotly interactives**. Il faut un
navigateur :

```bash
# depuis la racine du repo
xdg-open for_colleagues/figures/s4_HERO_choropleth_interactive.html
xdg-open for_colleagues/figures/s5_HERO_irf_animation.html
```

Ou simplement double-cliquer dessus dans l'explorateur de fichiers. **Ne pas** les
ouvrir dans le preview de VS Code — Plotly ne se charge pas là. Elles sont conçues
pour la soutenance orale (à projeter depuis un laptop) et comme bons compléments à
partager en ligne.

## Ce qui a changé au niveau du projet

- Ajout de **`src/course/`** (5 modules : descriptives, series_properties,
  granger_country, group_compare, var_irf) et de **`src/course/style.py`** (style
  matplotlib/Plotly unifié).
- Ajout des entrées **`pyproject.toml`** pour `plotly` et `kaleido` (faire `uv sync`
  si vous pullez).
- Ajout de **`REPORT_COURSE.md`** — la rédaction complète des 5 sections que vous
  pouvez utiliser comme source de texte.
- Ajout de **`DELIVERABLES.md`** — le changelog technique long (ce fichier en est la
  version courte).
- Le rapport méthodes-paper original (`REPORT.md`) et son code sont intacts. Rien n'a
  été supprimé.

## Si une figure a besoin d'un ajustement

Chaque module fait ~150 lignes et est autosuffisant. Exemples :

- **Changer les couleurs** → éditer `src/course/style.py`, dictionnaires
  `GROUP_COLORS_DOC` et `CAT_COLORS_DOC`.
- **Changer la taille** → éditer `FULL_PAGE`, `TWO_PANEL`, `SQUARE`, etc. dans le
  même fichier.
- **Changer le groupe d'un pays** → éditer `OECD_ISO3` ou `NORTH_AFRICA_ISO3` dans
  `src/course/groups.py`.
- **Tout regénérer** :
  ```bash
  for m in descriptives series_properties granger_country group_compare var_irf; do
      .venv/bin/python -m src.course.$m
  done
  ```
  Temps total ~90 s.

Ensuite il suffit de re-copier depuis `outputs/figures/course_*` dans ce dossier (ou
de pointer directement le rapport vers les versions dans `outputs/figures/`).

## Méthodologie en 30 secondes (pour l'oral)

- 128 pays, 1990–2023, panel V-Dem `v2x_corr` × SWIID `gini_disp`.
- 3 groupes : OCDE (38), Émergents (56), Afrique = ASS + Maghreb (34).
- Tests ADF + KPSS croisés (verdict stationnaire = ADF rejette ET KPSS ne rejette pas)
  → la majorité non-stationnaire en niveaux → **différenciation d'ordre 1**.
- Granger pays-par-pays sur séries différenciées, lags 1 et 2, plus petite p-value
  retenue (α effectif ≈ 9.75 %).
- Classification 4 catégories par pays. Choroplèthe = vue géographique.
- VAR bivarié par groupe avec ordre AIC ≤ 2 ; IRF Cholesky avec corruption en
  premier ; IC 90 % asymptotiques.

## Résultat headline

> Sur les 128 pays, **80 % ne montrent aucune causalité Granger détectable**. Les 22
> pays restants sont presque équirépartis entre C→I (10) et I→C (10), avec 2 cas
> bidirectionnels. **L'effet C→I n'est robuste qu'au sein du groupe Émergents** :
> leur VAR bivarié donne une IRF positive et significative (IC 90 % au-dessus de
> zéro pour h = 0..4), tandis que l'OCDE et l'Afrique sont à zéro. La conclusion
> centrale : **la causalité corruption → inégalités est forte dans les économies en
> transition, faible ou absente ailleurs**.
