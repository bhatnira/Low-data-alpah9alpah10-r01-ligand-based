# Selectivity panel expansion drop-in directory

Drop additional non-alpha9alpha10 nAChR modulator CSVs here to *extend* the
selectivity / decoys panel. No code changes needed — re-run the phase:

    PYTHONPATH=<project> python3 <project>/src/other_modulators.py
    # or: python3 <project>/workflows/run_all.py --only selectivity

## Required columns (same schema as `<repo>/otherModulators.csv`)

| Column        | Type  | Meaning                                                    |
|---------------|-------|------------------------------------------------------------|
| `File Name`   | str   | unique compound / series identifier (e.g. `PNU-120596.mol`)|
| `cleanedMol`  | str   | SMILES (canonicalized automatically)                       |
| `classLabel`  | num   | within-source binary class (1.0 / 0.0). NOT an a9a10 assay result |

Extra columns are tolerated and ignored.

## Merge rules

- Files are merged in deterministic order (repo-root `otherModulators.csv`
  first, then `*.csv` here sorted by name, then any files listed under
  `config.yaml -> data -> selectivity_panel -> sources`).
- Rows with identical **canonical** SMILES are deduplicated; the first-seen
  source wins. Dedup count and per-source counts are recorded in
  `selectivity/selectivity_manifest.json`.
- Each source file is versioned by content sha256 in the manifest.

## Isolation contract

- The panel is stored under `<project>/selectivity/` only.
- It is **never** written into `data/processed/` and never merged into the
  alpha9alpha10 training set.
- An isolation guard reports exact-SMILES overlap with the alpha9alpha10
  training set on every run (`isolation` block in the manifest).