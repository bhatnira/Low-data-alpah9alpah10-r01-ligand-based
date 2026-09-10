# RESEARCH STRATEGY
# (Significance, Innovation, Approach)

## A. SIGNIFICANCE

**Disease relevance.** alpha9alpha10 nicotinic acetylcholine receptors mediate
efferent innervation of the cochlea and are implicated in chronic pain, tinnitus,
and noise-induced hearing loss. Genetic deletion renders animals resistant to
several pain models, making this receptor a compelling non-opioid pain target and
an otoprotection target. Therapeutic translation requires positive allosteric
modulators (PAMs) that do not duplicate the toxicity profile of nicotinic
agonists.

**Chemical gap.** Known nicotinic modulators (agonists/antagonists of alpha4beta2,
alpha7) are based on amine and quaternary chemistries that do not transfer to
alpha9alpha10 PAM pharmacology. There is no experimentally enumerated PAM
chemotype library. Our preliminary data: a curated internal dataset of 30
validated compounds yields 7 PAM actives spanning 17 scaffolds/series; a
computational panel reaches leave-one-compound-out balanced accuracy 0.77
(ridge descriptors) and series/scaffold-out 0.80 (gradient boosting with
fingerprints + descriptors). These metrics are honest but modest - exactly the
situation that motivates empirical enrichment rather than extrapolation.

**What is missing (and is addressed here).** (1) A drug-like, chemically diverse
library: the current 15-compound candidate portfolio (9 chemistry-validated
analogs + 6 negative controls; 97 naive analog SMILES were enumerated but 86
failed chemistry validation - multi-fragment outputs - and were excluded,
recorded in `medchem_rejected_invalid.csv`), after
pre-registered PAINS/BRENK/Lipinski filtering and duplicate removal, contains
only 6 unique drug-like synthetically-credible molecules - a finding the field
needs to see,
and the starting point for medicinal-chemistry fill-in supplemented by
generative AI. (2) A demonstration
that the workflow enriches SAR: no experimental round has yet been run. (3) An
enriched, reusable SAR resource: current SAR products (56 matched-molecular-pair
transforms, 3-source TAF-3 hypothesis) are computational; they must be turned
into experimentally confirmed, AI-ready rules.

## B. INNOVATION

1. **Comparator-arm closed-loop design.** Rather than trusting model rankings,
   every round randomly splits an equal-size diversity-matched control arm,
   granting an internal, causal estimate of whether model-guided selection beats
   chemical diversity. Pre-registered before any data land.
2. **Honesty as method.** Constant/uninformative explanation methods are flagged
   rather than promoted; non-significant permutation nulls are reported; the
   last experimental round is held out of every refit; power is recomputed from
   pooled variability with an explicit prohibition on post-hoc re-powering.
3. **Drug-discovery-ready filtering inside the loop.** PAINS/BRENK/Lipinski
   gates and scaffold caps run inside plate design, converting a raw virtual
   portfolio into a curated chemotype library deliverable.
4. **Stereochemistry made testable.** Enantiomer/diastereomer probes are
   engineered into plates (22 designed from the 7 actives) so stereo hypotheses
   become experimentally distinguishable rather than 2D counterfactuals only.
5. **Selectivity as a measured design axis (phase 4b).** A curated, isolated panel
   of 70 non-alpha9alpha10 nAChR modulators with an enforced training-isolation
   contract is scored by the frozen model at design time (baseline: mean
   predicted alpha9alpha10 activity 0.215 +/- 0.045; 0/70 above the 0.5
   threshold); cross-subtype potential becomes an explicit, re-evaluated filter
   inside the loop instead of an afterthought.
6. **Dual focus: systematic ascorbate SAR + cochlear-delivery-compatible
   design.** The aim is innovative in its dual focus: (1) systematic SAR of
   ascorbate to define structural requirements for alpha9alpha10 PAM activity,
   and (2) strategic design of transporter-compatible analogs to facilitate
   targeted cochlear delivery. The use of advanced synthetic methodologies
   (deoxygenation, olefination, azalactam and rigid bicyclic construction)
   combined with mechanistic design enhances the potential for developing
   dual-acting compounds that are both potent PAMs and deliverable to the
   cochlea. Because we have already succeeded in identifying a significantly
   more potent analog of ascorbic acid (ID12, 0.198 uM; ID25, 2.63 uM), the
   success of this aim appears likely - and is tested empirically in the
   pre-registered rounds.
7. **Route-prediction decision support (phase 4c).** Retrosynthetic route
   planning (AIZynthFinder, public USPTO templates + ZINC stock) is a live
   pipeline phase that appends PREDICTED_ROUTE/UNVERIFIED plans and a
   per-building-block stock screen to every plate candidate, giving the
   oversight gate's synthesizability review concrete, reproducible, honest
   decision support (including explicit NO_ROUTE_FOUND reporting) instead of
   a hand-waved assumption.

## C. APPROACH

### Preliminary data (all verified in this repository; 19-phase pipeline runs end-to-end, all phases pass).
- Dataset governance: 30 compounds; 7 actives (IDs 1,2,3,12,18,24,25);
  24/30 share the ascorbate gamma-lactone enediol core (SMARTS
  OC1=C(O)C(=O)OC1-[#6]); 17 scaffolds/series; 23 tautomers normalized;
  stereochemistry audit completed.
- Ascorbate SAR sub-series (phase 3b, novel finding). Reference L-ascorbic acid
  (ID1, 1797 uM, 286%) and its D-form isoascorbate (ID2, 1316 uM, 300%; PI-note
  L/D assignment — stereo undefined in source) both potentiate with similar
  potency, i.e. ascorbate is a stereochemically tolerant pharmacophore and a
  selective alpha9alpha10 potentiator (PI note; not yet modeled). Position-
  resolved annotation (2-O vs 3-O, standard ascorbate numbering):
  * 3-O-alkoxy, 5,6-acetonide: 8 compounds (IDs 5-12), 1 active (ID12),
    0.198 uM = MOST POTENT in the set (~9,100x below the 1797 uM reference;
    PI shorthand: 3-O-ethyl arm)
  * 6-bromo-6-deoxy: 1 compound (ID25), 1 active, 2.63 uM (second most potent;
    ~680x gain over reference L-ascorbate)
  * 6-O-benzyl ether: 1 compound (ID18), 1 active, 1288 uM
  * 2-O-alkoxy: 1 compound (ID24), 1 active, 1202 uM
  * Free ascorbate: 3 compounds (IDs 1, 2, 4), 2 actives (66.7% — the two
    reference stereoisomers)
  * 5,6-acetonide, free enediol: 1 compound (ID3), 1 active, 6077 uM
  * 2,3-bis-O-alkoxy (open or acetonide): 6 compounds (IDs 22, 13-17),
    0 actives
  * 3-O-alkoxy, open chain: 3 compounds (IDs 19-21), 0 actives
  * Non-ascorbate: 6 compounds (IDs 23, 26-30), 0 actives
  Pharmacophore envelope: 3-O-alkylation is compatible with activity and, with
  a small 3-O-substituent plus 5,6-acetonide, defines the most potent compound;
  2-O-alkylation is tolerated but weak; 2,3-bis-O-alkylation ablates activity;
  C6 halogenation (6-bromo-6-deoxy) delivers the sharpest single-edit potency
  gain over the reference.
  Potency headroom: the validated reference is millimolar (1797 uM), yet
  sub-series edits reach 0.198 uM (3-O-alkoxy plus 5,6-acetonide, ~9,100x
  below reference) and 2.63 uM (6-bromo-6-deoxy, ~680x) — i.e., substitution
  edits, not the enediol core itself, set potency, bringing a validated
  vitamin-C pharmacophore into the drug-relevant sub-micromolar window.
- 56 ascorbate-restricted MMP transforms (both arms on core).
- Candidate-panel ascorbate linkage: 8/15 prospective candidates preserve the
  ascorbate core; all 15 carry the open enediol motif (tentative).
- Validation: LOCO best 0.77 (ridge, descriptors); scaffold/series-out 0.80
  (GBM, fingerprints+descriptors); bootstrap CIs computed; permutation nulls
  not significant -> importances treated as corroborative only.
- Explanations: across-model/seed stability STABLE (Spearman 0.88-0.89) but
  cross-method agreement UNSTABLE (~0.2) - motivates consensus fusion + flagging.
- SAR: 56 MMP transforms; the 6-bromo-6-deoxy edit (free ascorbate 1797 uM ->
  ID25 2.63 uM, ~680x) is the largest activity cliff and guides C6-halogen
  focus; TAF-3 only MODERATE (3 corroborative in-silico sources);
  TAF-1/2/4/6 LIMITED.
- Selectivity panel (phase 4b): 70 unique non-alpha9alpha10 nAChR modulators
  curated with an enforced isolation contract (0 exact-SMILES and 0 scaffold
  overlap with alpha9alpha10 training; stored under `selectivity/` only, never
  merged into training labels). Frozen-model baseline: mean predicted alpha9alpha10
  activity 0.215 +/- 0.045; 0/70 above the 0.5 threshold (alpha9alpha10 actives
  score >=0.64, mean 0.797). Panel is expandable via drop-in CSVs without code
  changes.
- Library projection: 8 Bemis-Murcko-distinct chemotype families (tiers: 2
  tier-2 hypothesis-group families, 6 tier-3 residual); 7 novel scaffolds vs
  training; after
  PAINS/BRENK/Lipinski + dedup only 6/15 unique molecules
  eligible -> defines realistic round size and the generative fill-in need.
- Closed-loop phase implemented and verified: consensus explanations, stereo
  probes, comparator-arm plate design, oversight gate (blocks 100% until
  med-chem sign-off), pre-registration file per round, append-only pooling,
  versioned pooled refits (`lbm_rf_fp_desc_rN.joblib`), power-from-pool,
  hit-metrics uploader, generator/structural data watch.
- Route-prediction decision support (phase 4c, NEW): AIZynthFinder retrosynthetic
  route planning (public USPTO expansion/templates + ZINC stock, Python 3.12
  venv worker) appends PREDICTED_ROUTE/UNVERIFIED plans and a per-building-block
  stock screen to round candidates (1/8 round-1 targets has a PREDICTED_ROUTE,
  1-step with the route product verified against the target; 18/38 building
  blocks IN_STOCK in ZINC; explicit NO_ROUTE_FOUND for the other 7, with recorded
  probabilities, never fabricated). The oversight gate's synthesizability
  criterion is now reviewable against this reproducible evidence; the phase
  self-reports NOT AVAILABLE if models/venv are absent. See
  `src/synth_route.py`, `synth_route/synth_route_report.md`.
- AF3 binding model: in validation phase (co-PI laboratory); ingestion hook
  implemented in `src/structures.py`; will provide TAF-6 binding-mode features
  when validated.

### Aim 1 (single integrated aim) - Medicinal-chemistry-driven analog program
### with computational/AI support

*Design.* Take the 15-compound portfolio (groups A-F across 32 assignment
rows: 1/6/1/5/12/7; unique molecules: 9 analogs + 6 negative controls). Apply,
in order: (i) deduplicate by molecule; (ii) largest-fragment canonical SMILES +
Bemis-Murcko scaffold assignment; (iii) PAINS filter, BRENK reactive-substructure
filter, Lipinski gate (MW<=500, cLogP<=5, HBD<=5, HBA<=10, <=2 violations);
(iv) tier assignment (1 = novel scaffold + clean; 2 = hypothesis-group member;
3 = residual); (v) scaffold cap of 2 per family per plate.
*Deliverables.* `chemotype_families.csv` (families, tiers, representative,
priority), `chemotype_library_annotated.csv`, `coverage_targets.csv`
(pre-registered per-round new-chemotype targets, EF threshold, go/no-go),
stereo probe set (22 probes from 7 actives), per-round route + stock
annotations (phase 4c).
*Fill-in (activity 1a, computational as support).* Because only 6 drug-like
molecules pass, Aim 1 includes two honest fill-in strategies: (a) expert
medicinal-chemistry expansion of the clean families by MMP-guided analog
enumeration around confirmed actives - the primary driver of analog supply;
(b) REINVENT4 Mol2Mol generation (stereo-aware prior) as computational
supplement - currently NO molecules exist because the binary is absent; any
generated molecules must pass the same filters and are reported with
provenance, not fabricated.

*Integrated synthesis program (Studies 1.1-1.5).* All are planned, not-yet-
executed synthesis programs (PENDING_SYNTHESIS; analog IDs are planning
labels, not fabricated structures). Each study tests a defined structural
feature; every resulting analog must pass the PAINS/BRENK/Lipinski gates and
is reported with provenance.

- Study 1.1 (role of the C-2 hydroxyl): the dataset shows C-2 mono-alkylation
  is tolerated but weak (ID24, 2-O-alkoxy, 1202 uM) while 2,3-bis-O-alkylation
  ablates activity (0/6). A C-2 edit series — 2-deoxy-2-halo (F, Cl, Br),
  2-O-acyl, 2-O-alkyl preserving the C-3 OH, plus C-2 epimers — tests whether
  C-2 substitution is tolerated, enhances, or blocks potentiation with the
  smallest possible perturbation to the enediol core.
- Study 1.2 (conformationally rigid bicyclic ascorbate analogs): constrain the
  C-5/C-6 side chain and enediol geometry into fused bicyclic frameworks
  (5,6-acetonide and cyclopentano/oxabicyclic locks around the gamma-lactone)
  to pre-organize the pharmacophore in its presumed active conformation and
  decouple side-chain rigidity from substitution at C-3.
- Study 1.3 (C-3-O-heterocyclic analogs): the C-3-O position is the
  highest-potency vector (ID12, 3-O-propargyl-5,6-acetonide, 0.198 uM) but
  only 1/8 other C-3-O-alkoxy acetonides are active, so the SAR at C-3 is
  narrow. O-linked azoles/azines/oxetanes and small heteroarylalkyl
  substituents (triazoles, pyridylmethyl, oxetanylmethyl,
  tetrahydrofuranylmethyl) plus heteroatom linkers map the exact steric/
  electronic envelope to make C-3 a reliable potency handle.
- Study 1.4 (C-5/C-6 probe): a fully deoxygenated analog (analog 4.1) is
  planned by Barton-McCombie deoxygenation (C-3 hydroxyl protected via
  Mitsunobu; C-5/C-6 hydroxyls activated as di-imidazole thiocarbonates; radical
  removal of both oxygens). Expected lower hydrophilicity; predicted reduced
  potency is a HYPOTHESIS. Verifiable precedent is limited: the only C-5/C-6
  blockade in the dataset (5,6-acetonide, ID3) is ~3.4x less potent than the
  reference (6077 vs 1797 uM), consistent with reduced activity on C-5/C-6
  derivatization - but the C-6 bromide (ID25, 2.63 uM) is a counter-example, so
  the prediction must be compound-specific, not blanket. A separate elimination/
  dehydration route is planned (NOT deoxygenation, which removes O to give C-H
  and does not by itself create an alkene) to afford a rigid C-5/C-6 olefinic
  handle for ruthenium-catalyzed cross-metathesis, yielding an olefinated
  analog series (e.g., analog 4.3) bearing varied R5 groups to probe steric and
  electronic effects. ~10 analogs planned across this study. These analogs also
  serve as platforms for further C-2/C-3 modification.
- Study 1.5 (deoxy-ascorbate mimics and azalactams): L-serine-derived chiral
  amino-butenolide (Dutta lactone) is planned to give azacarbohydrates (5.2) via
  rapid intramolecular cyclization; the unsaturated lactone is then planned for a
  Pd-catalyzed cyclopropanation (5.3a), in-situ peroxy epoxidation (5.3b), or
  in-situ N-mediated aziridination (5.3c) to afford bicyclic lactones that mimic
  ascorbate with distinct stereoelectronics (oxygen replaced/modulated by CH2 or
  N). ~8 analogs planned per category (cyclopropane/epoxide/aziridine), as
  candidacy for deoxy-ascorbate and azalactam chemotypes.
These studies test the stereoelectronic hypothesis that the C2=C3 enediol
geometry (rigidity and directionality of the hydroxyls) and C-5/C-6 oxygen
context set PAM activity. If a planned rigid-bicyclic/azalactam route fails,
the analog is replaced by a synthetically tractable scaffold-matched
alternative and the substitution is logged at the oversight gate, never
silently dropped.
*Pitfalls/alternatives.* Over-filtering to zero molecules: we retain filtered
molecules in a flagged "SAR-probe" tier for med-chem sign-off, so chemistry is
not discarded, only de-prioritized. Phosphonate-containing series (BRENK-flagged)
are explicitly reviewed by the medicinal chemist because they may be functionally
relevant to this target.

### Iterative closed-loop analog rounds (activity 1c)
*Design.* Per round: drug-like compounds from the eligible set (round 1 = 6,
currently BLOCKED pending med-chem sign-off). Arms: model-ranked (top by an openly documented rank score =
predicted activity x information-gain boost, scaffold-capped) vs
diversity-randomized (scaffold-capped random control). All blinded
(`R{n}-{crc32}` IDs). Oversight gate: medicinal-chemistry sign-off on
synthesizability, stereo feasibility, assay suitability, TAF consistency, and
negative-control balance; no experiment proceeds before approval. The
synthesizability criterion is concretized by route-prediction decision support
(phase 4c): each candidate's plate record carries PREDICTED_ROUTE/UNVERIFIED
retrosynthetic plans + building-block stock status, so the gate reviews
reproducible evidence (including NO_ROUTE_FOUND) instead of an assumption.
Pre-registered
prior to data: primary endpoint, EF threshold 2.0, BH-FDR alpha 0.10,
confirmation by independent dose-response, go/no-go (continue iff EF>=2.0 and
>=1 clean novel chemotype confirmed; max 3 rounds).
*Methods.* Two-electrode voltage-clamp on commercially sourced Xenopus oocytes;
concentration-response for potentiation on alpha9alpha10. Each compound ≥3
oocytes, blind operator to arm. Activity = potentiation>0 AND potency>0.
*Analysis (pre-registered).* EF_topk = (hits_topk/k)/(hits_total/n) per arm;
arm-stratified hit rates; precision@k; BH-FDR across compounds when replicate
p-values are available; enrichment factor vs pre-registered threshold.
*Model update after each round.* Append-only pooling (never duplicate/overwrite);
refit RandomForest (200 trees, Morgan r2 2048 + 10 descriptors, StandardScaler)
with the LAST round held out; versioned artifact
`models/classical/lbm_rf_fp_desc_rN.joblib`; power recomputed from pooled
variability (no post-hoc re-powering). TAF evidence updated only from confirmed
hits, with source counts tracked.
*Power/caveat.* With ~9-15 per round, individual-round statistics are weak; the
statistical unit is the pooled comparison across all rounds, which is why EF is
evaluated on pooled top-k and why the primary readout is the comparator-arm
contrast, not per-compound p-values.
*Selectivity filter inside the loop.* Before plate locking, the frozen model's
selectivity panel score (mean predicted alpha9alpha10 activity across the 70
non-alpha9alpha10 modulators) and each candidate's structural overlap with panel
scaffolds are re-evaluated and reported in the pre-registration file, so
cross-subtype potential is a documented, re-checked design axis.
*Pitfalls/alternatives.* Low eligible pool -> adaptive round size (already
implemented). If EF is below threshold in 3 rounds, the go/no-go fires a failure
report - a publishable negative that prevents overfitting to a small dataset.

### Predictive model construction + external validation and SAR packaging
### (activity 1d -> deliverable)
*Design.* (1) Predictive model after each round (append-only pooling, LAST round
held out, refit RandomForest); (2) External validation: assemble published
modulator data for
alpha9, alpha10, and nearest alpha subunits (ChEMBL cross-reference) to create
an external holdout; enriched models must show balanced accuracy above the
naive baseline on external data before being used for confirmatory ranking.
Currently MARKED NOT AVAILABLE with the framework in place
(`external_validation_status.json`); we commit to populating it from public
data within YP1; no claim is made until done. (2) SAR packaging: generate
per-chemotype structure-activity cards, MMP transform rules with activity
deltas, versioned model artifacts + hashes, and an AI-ready data export with
provenance for each row. (3) Selectivity packaging: the 70-compound panel,
isolation manifest, and model-score table are versioned alongside the other
deliverables. (4) All claims traceable: each SAR claim links to raw
data file, code, and seed via `reports/index.html` and the cycle ledger.
*Pitfalls/alternatives.* External data may be sparse/heterogeneous (different
assays, subunits) -> publish an honest harmonization report; keep internal
holdout as the primary reproducibility guard; never merge external activity
labels into training without a governing audit.

### Timeline and milestones
- Year 1: complete SAR-guided analog enumeration (Aim 1a), ascorbate sub-series
  SAR enrichment (phase 3b), generate fill-in chemotypes, AF3 binding-model
  validation (co-PI), run round 1 (design, route/stock annotation, gate, TEVC),
  ingest + refit v1 predictive model.
- Year 2: rounds 2-3, pooled refit v2-v3, EF evaluation, go/no-go, external
  data assembly (predictive-model validation start), AF3 structure-guided SAR
  integration (TAF-6).
- Year 3: predictive-model external validation + release, AI-ready enriched-SAR
  export (including ascorbate sub-series cards and selectivity benchmark),
  deposition (resource sharing plan), final report;
  failure-report pathway if EF<2.0.

### Rigor and reproducibility (summary)
Seeds fixed and recorded (global 42; per-module); append-only data; versioned
artifacts; pre-registration files frozen before experiments; leakage audit and
18-question final audit run at every pipeline pass; overclaim-scan identifies
unsupported language automatically; human-subject work: not applicable.