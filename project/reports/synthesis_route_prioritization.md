# SYNTHETIC-ROUTE PRIORITIZATION (Phase 13c)

> Honest, evidence-typed decision-support. Reaction classes are **derived from the real RDKit substructure** of each designed molecule; feasibility scores re-read from the experimental portfolio `synthetic_feasibility` column. **No literature routes or yields are invented.** Nothing here claims a compound WILL synthesize.

* in-domain Tier2 cohort: 17 compounds

* portfolio synthetic_feasibility distribution:

    - FAVORABLE  11

    - MODERATE   6

    - CHALLENGING 0


## Priority verdicts


| Molecule | stereo | detected reaction classes | feas score | priority |
| -------- | ------ | ------------------------- | ---------- | -------- |

| REV-INFORMATION_GAIN-9369 | none-det | N-ALKYLATION; STEREOCENTER_PRESENT | 0.80 | A_SYNTHESIZE |

| REV-INFORMATION_GAIN-7288 | none-det | STEREOCENTER_PRESENT | 0.60 | B_SYNTHESIZE_IF_CAPACITY |

| REV-INFORMATION_GAIN-7233 | PRESERVE | ACETAL_FORMATION; AMIDE_COUPLING;  | 0.60 | B_SYNTHESIZE_IF_CAPACITY |

| REV-INFORMATION_GAIN-9763 | none-det | STEREOCENTER_PRESENT | 0.80 | A_SYNTHESIZE |

| REV-INFORMATION_GAIN-12222 | none-det | FRIEDEL_CRAFTS_ACYLATION; REDUCTIO | 0.80 | A_SYNTHESIZE |

| REV-INFORMATION_GAIN-8365 | none-det | STEREOCENTER_PRESENT | 0.80 | A_SYNTHESIZE |

| REV-INFORMATION_GAIN-7489 | none-det | STEREOCENTER_PRESENT | 0.80 | A_SYNTHESIZE |

| REV-INFORMATION_GAIN-8270 | none-det | STEREOCENTER_PRESENT | 0.60 | B_SYNTHESIZE_IF_CAPACITY |

| REV-INFORMATION_GAIN-8727 | none-det | STEREOCENTER_PRESENT | 0.60 | B_SYNTHESIZE_IF_CAPACITY |

| REV-INFORMATION_GAIN-11135 | none-det | STEREOCENTER_PRESENT | 0.80 | A_SYNTHESIZE |

| REV-INFORMATION_GAIN-10130 | none-det | AMIDE_COUPLING; N-ALKYLATION; REDU | 0.80 | A_SYNTHESIZE |

| REV-INFORMATION_GAIN-10605 | none-det | N-ALKYLATION; STEREOCENTER_PRESENT | 0.80 | A_SYNTHESIZE |

| REV-INFORMATION_GAIN-6609 | none-det | N-ALKYLATION; STEREOCENTER_PRESENT | 0.60 | B_SYNTHESIZE_IF_CAPACITY |

| REV-INFORMATION_GAIN-8661 | PRESERVE | AMIDE_COUPLING; N-ALKYLATION; REDU | 0.80 | A_SYNTHESIZE |

| REV-INFORMATION_GAIN-11797 | none-det | O-ALKYLATION; STEREOCENTER_PRESENT | 0.80 | A_SYNTHESIZE |

| REV-INFORMATION_GAIN-11944 | PRESERVE | STEREOCENTER_PRESENT | 0.80 | A_SYNTHESIZE |

| REV-INFORMATION_GAIN-6940 | none-det | STEREOCENTER_PRESENT | 0.60 | B_SYNTHESIZE_IF_CAPACITY |


## Per-compound route basis (first 4)


### REV-INFORMATION_GAIN-9369

* SMILES: `Clc1sc(Cl)c2c1CCNC2`

* feasibility: FAVORABLE (score 0.80)

* ascorbate vulnerability: NONE_detected

* routes:


| route | lead transformation | composite score |
| ----- | ------------------- | --------------- |

| Route 1 | N-ALKYLATION | 0.720 |

| Route 2 | STEREOCENTER_PRESENT | 0.480 |



### REV-INFORMATION_GAIN-7288

* SMILES: `[B-][N+](C)(C)C`

* feasibility: MODERATE (score 0.60)

* ascorbate vulnerability: NONE_detected

* routes:


| route | lead transformation | composite score |
| ----- | ------------------- | --------------- |

| Route 1 | STEREOCENTER_PRESENT | 0.360 |



### REV-INFORMATION_GAIN-7233

* SMILES: `C/C=C/C=C/C(=O)N1C(=O)O[C@H]2O[C@H](C)O[C@@H](C)[C@@H]21`

* feasibility: MODERATE (score 0.60)

* ascorbate vulnerability: NONE_detected

* routes:


| route | lead transformation | composite score |
| ----- | ------------------- | --------------- |

| Route 1 | AMIDE_COUPLING | 0.570 |

| Route 2 | ESTERIFICATION | 0.558 |

| Route 3 | HYDROLYSIS_ESTER | 0.540 |



### REV-INFORMATION_GAIN-9763

* SMILES: `BrC(Br)=C(Br)Br`

* feasibility: FAVORABLE (score 0.80)

* ascorbate vulnerability: NONE_detected

* routes:


| route | lead transformation | composite score |
| ----- | ------------------- | --------------- |

| Route 1 | STEREOCENTER_PRESENT | 0.480 |




---
*Generated end-to-end; every transformation plausibility needs a medicinal-chemist review before synthesis (§8).*
