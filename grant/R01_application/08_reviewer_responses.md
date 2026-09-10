# RESPONSE TO REVIEWER CRITIQUES (the "final upgrade")

This document answers the seven critiques directly and names where each is
implemented in the repo, the grant, and the runnable pipeline. It doubles as the
"reviewer-response" argument for the resubmission.

## C1. Zero experimental enrichment has occurred (`new_sar_relationships = 0.0`)
- **Answer:** The application is reframed as a pilot/plan: the single integrated
  Aim 1 (iterative analog rounds) will execute the first experiments. The entire
  enrichment mechanism (append-only pooling, pooled refits with last round held
  out, EF/FDR, power-from-pool) exists and is verified end-to-end in the
  pipeline.
- **Evidence:** `loop/hit_metrics_roundN` machinery + `refit_pooled` +
  `update_power_pooled`; `chemspace`'s `new_sar_relationships` metric is honest
  `0.0` today and is the primary output metric of the Aim-1 rounds.
- **Grant:** `02_research_strategy.md` (iterative rounds, Timeline);
  `01_specific_aims.md`.

## C2. n=30 / 7 actives cannot bear the inference load
- **Answer:** Models are reframed as rankers to be validated, not conclusions.
  Every ranking goes through external validation and comparator-arm
  testing (both part of the single integrated Aim 1). LOCO/scaffold-out numbers
  are reported with CIs and baselines; permutation nulls reported
  non-significant.
- **Evidence:** `validation/loco`, bootstrap CIs, permutation nulls, honesty
  flags throughout.
- **Grant:** `02_research_strategy.md` (Preliminary data; Rigor).

## C3. Mechanistic (TAF) framing outpaces evidence
- **Answer:** TAFs are now falsifiable hypotheses to be updated by confirmed
  hits; no mechanistic claim drives funding decisions. The single integrated
  Aim 1 is analog optimization + predictive model building; mechanism is a
  downstream product.
- **Evidence:** `taf/evidence/taf_evidence_matrix.csv` (MODERATE only for
  TAF-3); oversight gate requires TAF-consistency review.
- **Grant:** `01_specific_aims.md` (Aim 1); `02_research_strategy.md`.

## C4. No comparator arm - closed-loop benefit not testable
- **Answer:** FIXED IN CODE. Every plate now has a `model_ranked` arm and an
  equal-size `diversity_random` control arm (adapted to pool size; currently
  6/15 unique drug-eligibles), matched on chemotype tier, scaffold-capped;
  EF is evaluated arm-vs-arm and thresholded at 2.0.
- **Evidence:** `loop/round1_plate.csv` (`arm` column), pre-registration JSON
  (EF formula, threshold, BH-FDR alpha, go/no-go), `hit_metrics` uploader.
- **Grant:** `01_specific_aims.md` (Aim 1); `02_research_strategy.md`
  (iterative rounds).

## C5. Throughput (TEVC) vs 141-compound library mismatch
- **Answer:** Library claim bounded to what TEVC can test: 3 rounds x ~15
  compounds; candidate pool honestly shown to shrink to 6 unique drug-like
  molecules after filtering - the real round-1 set, with medicinal-chemistry
  fill-in (studies 1.1-1.5) as the primary analog supply and generative
  AI/computational proposals as a supplement - an explicit Aim-1 task rather
  than an unstated assumption.
- **Evidence:** `loop/library/chemotype_families.csv`; `coverage_targets.csv`;
  adaptive `per_round = min(per_round, len(eligible))` in `design_plate`.
- **Grant:** `02_research_strategy.md` (Aim 1 Pitfalls; Budget note).

## C6. Endpoint too narrow for drug-discovery SAR
- **Answer:** Added explicit secondary gates (PAINS/BRENK/Lipinski) inside the
  analog plates, and the Aim-1 deliverable packages per-chemotype SAR + MMP
  rules and an AI-ready export designed for downstream
  selectivity/physchem/preclinical use. A dedicated selectivity screen
  (phase 4b, NEW FILE `src/other_modulators.py`)
  curates 70 unique non-alpha9alpha10 nAChR modulators into an isolated panel
  (0 exact-SMILES and 0 scaffold overlap with alpha9alpha10 training; stored
  under `selectivity/`, never merged into training labels) and benchmarks the
  frozen model at design time (mean predicted alpha9alpha10 activity 0.215 +/-
  0.045; 0/70 above the 0.5 threshold), making cross-subtype selectivity an
  explicit, re-evaluated design axis inside the loop rather than an afterthought.
- **Evidence:** filters in `src/library.py` + `design_plate`; MMP export in
  `sar/mmp`; `src/other_modulators.py`; `selectivity/selectivity_report.md`.
- **Grant:** `01_specific_aims.md` (Aim 1); `02_research_strategy.md`; v2
  Innovation #5 and 3.3.4.

## C7. No statistical pre-specification / FDR / go-no-go
- **Answer:** FIXED IN CODE. `loop/pre_registration_round1.json` freezes before
  data: primary endpoint, EF formula + threshold 2.0, BH-FDR alpha 0.10,
  confirmation by independent dose-response, scaffold cap, eligibility filter,
  max rounds, and go/no-go. Not re-powered after results (explicit field:
  `no_post_hoc_repowering: true`).
- **Evidence:** `loop/pre_registration_round1.json`; `loop/hit_metrics_roundN`.
- **Grant:** `01_specific_aims.md`; `02_research_strategy.md` (Rigor).

## Additional self-critique (not from the prior review)
- **Duplicate molecule_ids in the portfolio** (32 assignments, 15 unique, see
  groups A-F): plates now dedupe by molecule; the finding is disclosed in the
  Research Strategy.
- **Non-unique blinding semantics** and **Xenopus as vertebrate work**: both
  addressed in `06_vertebrate_animals.md`.
- **No external data yet**: the external-validation step of the Aim-1
  predictive-model deliverable is explicitly reserved for it and the pipeline
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
  56 ascorbate-restricted MMP transforms; 8/15 prospective candidates preserve
  the core. This sub-series structure is the basis for Aim 1's ascorbate-
  pharmacophore envelope.
- **AF3 binding model (in validation phase):** co-PI laboratory is validating
  Alphafold3 binding models for the ascorbate-PAM series against alpha9alpha10.
  Ingestion hook implemented in `src/structures.py`; will provide TAF-6
  binding-mode features when validated. No structural claim made until done.
- **19-phase pipeline all OK:** audit, dataset, sar, sar_ascorbate, models, xai,
  taf, structures, medchem, reinvent, chemspace, candidates, library, lock,
  prospective, selectivity, synth_route, loop, reports.
- **Selectivity panel (new phase 4b, 19-phase pipeline):** the frozen model is
  benchmarked against 70 curated non-alpha9alpha10 modulators with an enforced
  isolation contract (0 exact-SMILES / 0 scaffold overlap with alpha9alpha10
  training; panel stored under `selectivity/` only). Baseline: mean predicted
  activity 0.215 +/- 0.045; 0/70 above the 0.5 threshold. Panel is expandable via
  drop-in CSVs in `data/selectivity/raw/` without code changes. See `src/
  other_modulators.py`, `selectivity/selectivity_report.md`, and
  `data/selectivity/raw/README.md`.
- **Synthesizability decision support (new phase 4c, 19-phase pipeline):**
  AIZynthFinder retrosynthetic route prediction (public USPTO templates + ZINC
  stock) runs as a standalone phase (`src/synth_route.py` +
  `src/aizynth_worker.py` in a Python 3.12 venv). It appends
  PREDICTED_ROUTE/UNVERIFIED plans + a per-building-block stock screen to every
  round candidate: 8 round-1 targets -> 1 plan (1-step, route product verified
  against the target), explicit NO_ROUTE_FOUND for the remaining 7; 18/38
  building blocks IN_STOCK in ZINC. This directly upgrades the oversight gate's
  "synthesizability" criterion (reviewer C5 concern) from a hand-waved
  assumption to reproducible decision support; the med-chem sign-off remains
  binding. Phase auto-skips and self-reports NOT AVAILABLE if models/venv are
  missing - no fabricated routes. See `synth_route/synth_route_report.md`.

---
### Where to look in the repo
- `src/loop.py` - design_plate (arms), pre_register_round, hit_metrics,
  refit_pooled, update_power_pooled, data_watch, cycle_ledger.
- `src/library.py` - chemotype families, filters, coverage targets, external-status.
- `src/sar_ascorbate.py` - ascorbate-core detection, sub-series SAR, MMP enrichment.
- `src/structures.py` - AF3/Boltz structural consensus (dormant until models validated).
- `src/reports.py` - Appendix G (closed-loop) + Appendix H (chemotype library) +
  Appendix I (ascorbate SAR enrichment).
- `src/synth_route.py` + `src/aizynth_worker.py` - phase 4c route-prediction
  decision support (PREDICTED_ROUTE/UNVERIFIED + stock screen).
- `workflows/run_all.py` - 19-phase e2e; all pass.
- `scientific_paper.txt` - technical report with limitation-mitigation table.