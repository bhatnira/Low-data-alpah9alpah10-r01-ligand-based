# RESPONSE TO REVIEWER CRITIQUES (the "final upgrade")

This document answers the seven critiques directly and names where each is
implemented in the repo, the grant, and the runnable pipeline. It doubles as the
"reviewer-response" argument for the resubmission.

## C1. Zero experimental enrichment has occurred (`new_sar_relationships = 0.0`)
- **Answer:** The application is reframed as a pilot/plan: Aim 2 will execute
  the first rounds. The entire enrichment mechanism (append-only pooling,
  pooled refits with last round held out, EF/FDR, power-from-pool) exists and is
  verified end-to-end in the pipeline.
- **Evidence:** `loop/hit_metrics_roundN` machinery + `refit_pooled` +
  `update_power_pooled`; `chemspace`'s `new_sar_relationships` metric is honest
  `0.0` today and is the primary output metric of Aim 2.
- **Grant:** `02_research_strategy.md` (Aim 2, Timeline); `01_specific_aims.md`.

## C2. n=30 / 7 actives cannot bear the inference load
- **Answer:** Models are reframed as rankers to be validated, not conclusions.
  Every ranking goes through external validation (Aim 3) and comparator-arm
  testing (Aim 2). LOCO/scaffold-out numbers are reported with CIs and
  baselines; permutation nulls reported non-significant.
- **Evidence:** `validation/loco`, bootstrap CIs, permutation nulls, honesty
  flags throughout.
- **Grant:** `02_research_strategy.md` (Preliminary data; Rigor).

## C3. Mechanistic (TAF) framing outpaces evidence
- **Answer:** TAFs are now falsifiable hypotheses to be updated by confirmed
  hits; no mechanistic claim drives funding decisions. Specific Aims are
  library + enrichment; mechanism is a downstream product.
- **Evidence:** `taf/evidence/taf_evidence_matrix.csv` (MODERATE only for
  TAF-3); oversight gate requires TAF-consistency review.
- **Grant:** `01_specific_aims.md` (Aim 2); `02_research_strategy.md`.

## C4. No comparator arm - closed-loop benefit not testable
- **Answer:** FIXED IN CODE. Every plate now has a `model_ranked` arm and an
  equal-size `diversity_random` control arm (adapted to pool size; currently
  9/99 unique drug-like molecules), matched on chemotype tier, scaffold-capped;
  EF is evaluated arm-vs-arm and thresholded at 2.0.
- **Evidence:** `loop/round1_plate.csv` (`arm` column), pre-registration JSON
  (EF formula, threshold, BH-FDR alpha, go/no-go), `hit_metrics` uploader.
- **Grant:** `01_specific_aims.md` (Aim 2); `02_research_strategy.md` (Aim 2).

## C5. Throughput (TEVC) vs 141-compound library mismatch
- **Answer:** Library claim bounded to what TEVC can test: 3 rounds x ~15
  compounds; candidate pool honestly shown to shrink to 9 unique drug-like
  molecules after filtering - the real round-1 set, plus generative/med-chem
  fill-in as an Aim-1 task rather than an unstated assumption.
- **Evidence:** `loop/library/chemotype_families.csv`; `coverage_targets.csv`;
  adaptive `per_round = min(per_round, len(eligible))` in `design_plate`.
- **Grant:** `02_research_strategy.md` (Aim 1 Pitfalls; Budget note).

## C6. Endpoint too narrow for drug-discovery SAR
- **Answer:** Added explicit secondary gates (PAINS/BRENK/Lipinski) inside the
  library plates, and Aim 3 packages per-chemotype SAR + MMP rules and an
  AI-ready export designed for downstream selectivity/physchem/preclinical
  use. A selectivity screen (vs other nAChR subtypes) is out of scope for this
  grant's experiments and is not silently assumed as an output.
- **Evidence:** filters in `src/library.py` + `design_plate`; MMP export in
  `sar/mmp`.
- **Grant:** `01_specific_aims.md` (Aim 3); `02_research_strategy.md`.

## C7. No statistical pre-specification / FDR / go-no-go
- **Answer:** FIXED IN CODE. `loop/pre_registration_round1.json` freezes before
  data: primary endpoint, EF formula + threshold 2.0, BH-FDR alpha 0.10,
  confirmation by independent dose-response, scaffold cap, eligibility filter,
  max rounds, and go/no-go. Not re-powered after results (explicit field:
  `no_post_hoc_repowering: true`).
- **Evidence:** `loop/pre_registration_round1.json`; `loop/hit_metrics_roundN`.
- **Grant:** `01_specific_aims.md`; `02_research_strategy.md` (Rigor).

## Additional self-critique (not from the prior review)
- **Duplicate molecule_ids in the portfolio** (141 rows, 99 unique): plates now
  dedupe by molecule; the finding is disclosed in the Research Strategy.
- **Non-unique blinding semantics** and **Xenopus as vertebrate work**: both
  addressed in `06_vertebrate_animals.md`.
- **No external data yet**: Aim 3 is explicitly reserved for it and the pipeline
  scans `data/external` honestly (`external_validation_status.json`).
- **Ascorbate SAR enrichment (new phase 3b):** The entire dataset is ascorbate-
  derived (24/30 share the gamma-lactone enediol core). Position-resolved
  sub-series SAR: L-ascorbic acid (ID1) is the reference and its D-form
  isoascorbate (ID2) potentiates with similar potency (PI-note L/D assignment);
  the 3-O-substituted arm is the high-potency direction (ID12, 3-O-propargyl-
  5,6-acetonide, 0.198 uM, ~9,100x below the 1797 uM reference), 6-bromo-6-deoxy
  (ID25, 2.63 uM) is second, 2-O-alkyl (ID24) and free-enediol acetonide (ID3)
  are >1 mM, and 2,3-bis-O-alkylation is inactive; non-ascorbate zero. Ascorbic acid is a selective
  alpha9alpha10 potentiator (PI note) and is not yet computationally modeled.
  56 ascorbate-restricted MMP transforms; 8/99 prospective candidates preserve
  the core. This sub-series structure is the basis for Aim 1's ascorbate-
  pharmacophore envelope.
- **AF3 binding model (in validation phase):** co-PI laboratory is validating
  Alphafold3 binding models for the ascorbate-PAM series against alpha9alpha10.
  Ingestion hook implemented in `src/structures.py`; will provide TAF-6
  binding-mode features when validated. No structural claim made until done.
- **17-phase pipeline all OK:** audit, dataset, sar, sar_ascorbate, models, xai,
  taf, structures, medchem, reinvent, chemspace, candidates, library, lock,
  prospective, loop, reports.

---
### Where to look in the repo
- `src/loop.py` - design_plate (arms), pre_register_round, hit_metrics,
  refit_pooled, update_power_pooled, data_watch, cycle_ledger.
- `src/library.py` - chemotype families, filters, coverage targets, external-status.
- `src/sar_ascorbate.py` - ascorbate-core detection, sub-series SAR, MMP enrichment.
- `src/structures.py` - AF3/Boltz structural consensus (dormant until models validated).
- `src/reports.py` - Appendix G (closed-loop) + Appendix H (chemotype library) +
  Appendix I (ascorbate SAR enrichment).
- `workflows/run_all.py` - 17-phase e2e; all pass.
- `scientific_paper.txt` - technical report with limitation-mitigation table.