# Non-alpha9alpha10 nAChR modulator selectivity panel

_Generated automatically by `project/src/other_modulators.py` (alpha9alpha10-nAChR-PAM-TAF)._

## 1. Motivation / data-integrity note

The compounds in `<repo>/otherModulators.csv` are modulators of **other** nAChR subtypes (alpha7 and alpha4beta2 PAMs, the CMPI/isoxazole-indole series, and nAChR-active flavonoids/natural products). They are **not** alpha9alpha10 measurements. To avoid corrupting the alpha9alpha10 `is_active` training target, they are kept as a separate, documented **selectivity / decoys panel** rather than merged into the training set. Uses:
- **Decoys / selectivity modeling:** negative or off-target signal for future selectivity-aware models and as a panel designed a9alpha10 hits should discriminate against.
- **Chemotype scan:** Bemis-Murcko scaffold overlap vs the ascorbate a9alpha10 library and candidate chemotype families.
- **Benchmark:** frozen a9alpha10 model predictions on the panel define the current selectivity baseline.

## 2. Panel summary

- Total panel: **70**; parseable: **70**
- Share with a9alpha10-training scaffolds: **0** (none)
- Overlaps with candidate chemotype families: **0**

## 3. Frozen a9alpha10 model on the panel (selectivity baseline)

- Mean predicted a9alpha10 PAM activity: **0.312** ± 0.057
- Fraction predicted active (>=0.5): **0.000**

> Interpretation: the frozen a9alpha10 model was trained only on the ascorbate chemistry. A low predicted-activity / low blacklisted fraction for this panel is the desired selectivity behaviour; a high fraction means designed hits could cross-react. The 7 a9alpha10 actives (ascorbate core) should separate clearly from this panel.

### Top-predicted panel compounds (potential selectivity watch-list)

| panel_id | predicted_a9a10 | novelty | scaffold_in_a9a10 | source_label | subtype_note |
|---|---|---|---|---|---|
| NS-206.mol | 0.495 | 0.658 | 0 | 1.0 | alpha7 PAM (benzoxanthin/quinoxalinone NS series) |
| 5,7-dihydroxy-4-phenylcoumarin.mol | 0.465 | 0.622 | 0 | 0.0 | flavonoid nAChR modulator |
| Genistein.mol | 0.440 | 0.709 | 0 | 0.0 | isoflavone nAChR modulator (natural product) |
| 5-hydroxyindole (5-HI).mol | 0.430 | 0.749 | 0 | 0.0 | alpha7 PAM (5-HI) |
| 6.mol | 0.430 | 0.786 | 0 | 0.0 | chalcone nAChR modulator (natural product) |
| Quercetin.mol | 0.415 | 0.718 | 0 | 0.0 | flavonoid nAChR modulator (natural product) |
| TBS-156.mol | 0.410 | 0.755 | 0 | 0.0 | alpha7 PAM (triazine-sulfonamide series) |
| 5.mol | 0.400 | 0.783 | 0 | 0.0 | chalcone nAChR modulator (natural product) |
| Isoliquirigenin.mol | 0.400 | 0.783 | 0 | 0.0 | chalcone nAChR modulator (natural product) |
| RGM079.mol | 0.400 | 0.768 | 0 | 0.0 | flavonoid nAChR modulator |
| NS-1738 (1).mol | 0.370 | 0.838 | 0 | 0.0 | alpha7 PAM (thiophene-urea) |
| Struc12.mol | 0.365 | 0.790 | 0 | 0.0 | indole PAM series (a7-type) |
| PAM-4.mol | 0.345 | 0.734 | 0 | 0.0 | alpha7 PAM (cinnamide NS-1738 analog) |
| Struc3.mol | 0.345 | 0.884 | 0 | 0.0 | indole PAM series (a7-type) |
| Com20.mol | 0.340 | 0.813 | 0 | 0.0 | indole PAM series (a7-type) |

## 4. Chemotype (Bemis-Murcko) scan vs ascorbate candidates

Overlap of other-subtype panel scaffolds with the a9alpha10 candidate chemotype families (a non-empty overlap flags a scaffold the alpha9alpha10 library shares with other-subtype modulators):

| family_id | a9alpha10 candidate chemotype | panel overlap count | overlapping panel IDs |
|---|---|---|---|

## 5. Physicochemical coverage of the panel relative to a9alpha10 library

Fraction of the other-subtype panel that falls inside the a9alpha10 training descriptor range (per descriptor, per a9alpha10 activity group).

| group | descriptor | train_min | train_max | frac_panel_in_train_range |
|---|---|---|---|---|
| a9a10_actives | MW | 176.124 | 266.249 | 0.271 |
| a9a10_actives | LogP | -1.407 | 0.483 | 0.000 |
| a9a10_actives | TPSA | 74.220 | 107.220 | 0.200 |
| a9a10_actives | HBA | 5.000 | 6.000 | 0.143 |
| a9a10_actives | RotBonds | 1.000 | 5.000 | 0.914 |
| a9a10_actives | RingCount | 1.000 | 2.000 | 0.500 |
| a9a10_actives | AromaticRings | 0.000 | 1.000 | 0.014 |
| a9a10_actives | FractionCSP3 | 0.308 | 0.667 | 0.529 |
| a9a10_inactives | MW | 148.114 | 396.439 | 0.886 |
| a9a10_inactives | LogP | -3.013 | 3.709 | 0.500 |
| a9a10_inactives | TPSA | 63.220 | 110.800 | 0.243 |
| a9a10_inactives | HBA | 5.000 | 8.000 | 0.157 |
| a9a10_inactives | RotBonds | 1.000 | 8.000 | 0.971 |
| a9a10_inactives | RingCount | 1.000 | 4.000 | 0.971 |
| a9a10_inactives | AromaticRings | 0.000 | 2.000 | 0.586 |
| a9a10_inactives | FractionCSP3 | 0.348 | 1.000 | 0.414 |

## 6. Isolation contract

- This panel is stored under `<project>/selectivity/` only. It is **never** written into `data/processed/` and the alpha9alpha10 training CSV (`modulator-dataset-a9a10.csv`) is never modified.
- An isolation guard compares canonical SMILES between the panel and the alpha9alpha10 training set on every run and reports any exact overlap in `selectivity/selectivity_manifest.json` (`isolation` block).

## 7. Expanding the panel later

The panel is deliberately small and can be extended without touching code:
- **Drop-in files:** add any number of CSV files to `<project>/data/selectivity/raw/` with the same columns as `otherModulators.csv` (`File Name, cleanedMol, classLabel`). Re-run the phase and they are merged automatically (each file is versioned by content hash in the manifest).
- **Config override:** optionally list extra files under `config.yaml -> data -> selectivity_panel -> sources`.
- **Deduplication:** rows with identical canonical SMILES are merged (first-seen source wins); per-source counts and the number deduped are recorded in the manifest.
- New chemotypes feed the same Bemis-Murcko scan and frozen-model selectivity benchmark; keep this section of the report as the living baseline.

## 8. Follow-up / honest caveats

- `source_label` is carried from the source file verbatim and is **not** an alpha9alpha10 measurement; treat only as a within-source binary class.
- Subtype notes are literature-informed annotations for interpretability; verify against primary references before publication.
- This panel has NOT been merged into the alpha9alpha10 training set by design.
