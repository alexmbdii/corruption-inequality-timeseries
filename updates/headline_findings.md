# Headline numbers — copy-paste into the report

Tous les chiffres ci-dessous sortent de notre pipeline et sont vérifiés contre les
CSV dans `for_colleagues/tables/`. Format prêt à coller en Word/Google Docs.

---

## §2 — Données et variables

### Échantillon
- **128 pays** (règle ≥ 20 années conjointes sur `v2x_corr` et `gini_disp`)
- **1990–2023** (fenêtre tests) ; **1996–2022** (fenêtre descriptive, couverture SWIID)
- **3 groupes** : OECD (38), Émergents (56), Afrique = SSF + Maghreb (34)

### Tableau résumé (à transcrire)

| Variable | Groupe | n pays | Moyenne | Écart-type | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| `v2x_corr` | OECD       | 38 | 0.13 | 0.16 | 0.00 | 0.85 |
| `v2x_corr` | Émergents  | 56 | 0.58 | 0.25 | 0.01 | 0.97 |
| `v2x_corr` | Afrique    | 34 | 0.61 | 0.21 | 0.09 | 0.97 |
| `gini_disp` | OECD      | 38 | 31.7 | 6.9  | 16.8 | 53.2 |
| `gini_disp` | Émergents | 56 | 39.1 | 6.9  | 21.2 | 54.8 |
| `gini_disp` | Afrique   | 34 | 45.2 | 7.4  | 32.4 | 65.2 |

> Lecture : l'OECD se distingue nettement — corruption ~5× plus basse, Gini ~13 points
> plus bas que l'Afrique. Émergents et Afrique sont proches sur la corruption mais
> l'Afrique reste 6 points plus haut sur le Gini.

---

## §3 — Tests de stationnarité (ADF + KPSS croisés)

Verdict « stationnaire » exige **ADF p < 0.05 ET KPSS p ≥ 0.05** (les deux tests sont
de sens opposé donc on croise).

| Variable | Groupe | n pays | Stationnaire | Non-stationnaire | Ambigu |
|---|---|---:|---:|---:|---:|
| `v2x_corr` | OECD       | 38 | 4 | 20 | 14 |
| `v2x_corr` | Émergents  | 56 | 6 | 31 | 19 |
| `v2x_corr` | Afrique    | 34 | 4 | 18 | 12 |
| `v2x_corr` | **TOTAL**  | **128** | **14** | **69** | **45** |
| `gini_disp` | OECD      | 38 | 7 | 21 | 10 |
| `gini_disp` | Émergents | 56 | 7 | 27 | 22 |
| `gini_disp` | Afrique   | 34 | 1 | 22 | 11 |
| `gini_disp` | **TOTAL** | **128** | **15** | **70** | **43** |

> **Verdict** : la **majorité des pays sont non-stationnaires en niveaux** (54% pour
> la corruption, 55% pour le Gini), 35–34% sont ambigus, seulement 11–12% sont
> clairement stationnaires. Conséquence : tous les tests de Granger sont appliqués
> sur **séries différenciées d'ordre 1** (Hard Rule cours).

---

## §4 — Granger pays-par-pays (sur séries différenciées)

Lags 1 et 2 testés ; on retient la plus petite p-value (α effectif ≈ 9.75% pour α brut
de 5%). Seuil de décision α = 5%.

### Classification 4 catégories

| Catégorie | OECD | Émergents | Afrique | **Total** | **% du panel** |
|---|---:|---:|---:|---:|---:|
| C → I uniquement   | 1  | 5  | 4  | **10**  | 8.0% |
| I → C uniquement   | 3  | 6  | 1  | **10**  | 8.0% |
| Bidirectionnel     | 0  | 1  | 1  | **2**   | 1.6% |
| Aucune causalité   | 32 | 44 | 27 | **103** | **82.4%** |
| n/a (T trop court) | 2  | 0  | 1  | 3       | 2.4% |

### Pays par catégorie (pour annexe ou commentaire textuel)

- **C → I (10 pays)** : BEL, EGY, FJI, KGZ, MLI, MYS, RUS, SWZ, UGA, YEM
- **I → C (10 pays)** : ALB, BRB, CHN, FRA, GIN, GRC, HKG, IRL, SRB, THA
- **Bidirectionnel (2 pays)** : BFA, VNM

> **Lecture** : 80% des pays n'ont pas de causalité Granger détectable au seuil 5%.
> Parmi les 22 qui en ont, la répartition C→I / I→C est presque symétrique (10 vs 10).
> **Pas de signal de pooling clair sur le panel mondial** — la direction de causalité
> est hautement spécifique au pays. Ceci rejoint Policardo et al. (2018).

### Lecture par groupe

- **Afrique** : part la plus haute de C→I (4/34 = **12%**) et la plus basse de I→C (1/34 = 3%)
- **OECD** : profil inverse — 1 C→I pour 3 I→C
- **Émergents** : équilibre 5 vs 6, plus le seul pays bidirectionnel (VNM)

> Cohérent avec la lecture de Sulemana & Kpienbaareh (2018) sur l'Afrique sub-saharienne.

---

## §5 — VAR bivarié par groupe + IRF (le résultat headline)

VAR(p) avec p sélectionné par AIC ≤ 2, sur séries pays-année différenciées et
démeanées par pays (effets fixes), Cholesky avec corruption ordonnée en premier.

| Groupe | n pays | n obs (différenciées) | Ordre VAR (AIC) |
|---|---:|---:|---:|
| OECD | 38 | 1 162 | 1 |
| Émergents | 56 | 1 585 | 2 |
| Afrique | 34 | 872 | 1 |

### IRF — choc Corruption → Inégalités, h = 0..4 (IC 90%)

| Groupe | h=0 mean [IC] | h=2 mean [IC] | h=4 mean [IC] |
|---|---|---|---|
| OECD       | −0.014 [−0.030, +0.002] | −0.003 [−0.013, +0.007] | −0.001 [−0.004, +0.002] |
| **Émergents** | **+0.020 [+0.009, +0.031]** | **+0.023 [+0.008, +0.039]** | **+0.013 [+0.003, +0.022]** |
| Afrique    | −0.005 [−0.012, +0.002] | −0.004 [−0.009, +0.001] | −0.001 [−0.003, +0.001] |

> **Le seul groupe avec un IRF C→I significativement positif** (IC 90% au-dessus de
> zéro pour h = 0..4) est **les émergents**. OECD et Afrique : zéro à tous les horizons.

### IRF — choc Inégalités → Corruption

Effet quasi nul partout (ordres de grandeur 10⁻³ à 10⁻⁴). Voir `s5_irf_static_panels.png`
(panneau bas-gauche).

---

## Conclusion en une phrase

> Sur 128 pays différents, la causalité Granger **corruption → inégalités** n'est ni
> globale (80% des pays sans signal détectable) ni absente : elle est **concentrée
> dans les économies émergentes**, où elle est positive et significative à 90%. Les
> économies développées (OECD) et l'Afrique ne montrent pas d'effet de groupe net,
> pour des raisons opposées (variations trop faibles vs chocs idiosyncratiques
> dominants).
