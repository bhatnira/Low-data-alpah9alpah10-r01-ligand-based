# SPECIFIC AIMS (1 page)

**Title:** Ascorbate-Derived Chemotype Library and SAR Enrichment for alpha9alpha10
Nicotinic Acetylcholine Receptor (nAChR) Positive Allosteric Modulators (PAMs)

alpha9alpha10 nAChRs are implicated in chronic pain, tinnitus, and cochlear
damage, yet no PAM chemotype has been translated, and publicly available
modulator chemistry fails to transfer to this rare subunit combination. Our
computational workflow (TAF = transferable activity functions) has curated 30
validated compounds (7 PAM actives), all sharing the L-ascorbic acid
(enediol-gamma-lactone) pharmacophore — the only validated PAM chemotype for
this target. We generated a 141-compound prospective candidate portfolio. The
gap is that (i) candidates are not curated into a drug-like, chemically diverse
chemotype library, (ii) the ascorbate SAR sub-series structure has not been
exploited, and (iii) no experimental round has demonstrated that the workflow
enriches SAR. Our long-term goal is an experimentally enumerated chemotype
library that enriches SAR for any future alpha9alpha10 drug-discovery or
preclinical program.

**Aim 1. Curate a diverse, ascorbate-derived alpha9alpha10 PAM chemotype library.**
Apply pre-registered structural filters (PAINS/BRENK reactive-substructure
removal, Lipinski gate, scaffold capping) to the 141-molecule portfolio to
deliver >=10 Bemis-Murcko-distinct chemotype families that are synthesizable and
assayable in TEVC. Characterize the ascorbate SAR sub-series around the
reference L-ascorbic acid (ID1, 1797 uM, 286% potentiation): its D-form
isoascorbate (ID2) potentiates with similar potency (1316 uM, 300%; PI-note
L/D assignment — stereo undefined in source); the 3-O-substituted arm is the
high-potency direction (ID12, 3-O-propargyl-5,6-acetonide, 0.198 uM, ~9,100x
below the 1797 uM reference; PI shorthand "3-O-ethyl"); 6-bromo-6-deoxy (ID25,
2.63 uM) is second-most-potent
(~680x over reference); 2-O-alkyl (ID24, ~1.2 mM) and free-enediol acetonide
(ID3, 6.1 mM) actives are >1 mM; and 2,3-bis-O-alkylation ablates activity
(0/6). Ascorbic acid is a selective alpha9alpha10 potentiator (PI note; not yet
computationally modeled). These characteristics define the SAR envelope for each
substitution pattern. Stereochemical probes (enantiomer/diastereomer) derived
from the 7 active parents will make stereo hypotheses experimentally
testable. Pre-register coverage targets and per-round go/no-go
criteria (implemented; current honest projection: 9/99 unique drug-like
molecules pass all filters, defining the realistic round-1 test set and the
amount of generative/med-chem fill-in needed). Pending alphafold3 binding-model
validation (in validation phase, co-PI laboratory), integrate structure-guided
enrichment as TAF-6 binding-mode features.

**Aim 2. Run a pre-registered, two-armed closed-loop experiment and show the
loop enriches SAR.**
In each round test ~15 drug-like compounds split into a model-ranked arm and an
equal-size diversity-randomized control arm (matched on chemotype tier),
blinded to the team. TEVC potentiation assays on alpha9alpha10-expressing
Xenopus oocytes (co-PI laboratory). Primary endpoint: potentiation>0 AND
potency>0, confirmed by independent dose-response. Pre-registered analyses:
enrichment factor (threshold 2.0), Benjamini-Hochberg FDR (alpha=0.10),
precision@k, arm-stratified hit rates. After each round, append results (never
overwrite), refit the model with the last round held out, update TAF evidence
and power from pooled variability (no post-hoc re-powering). Deliver at least 3
rounds / >=3 enriched-SAR releases, or a formal failure report if the
enrichment factor stays below threshold.

**Aim 3. Validate and package enriched SAR for future drug-discovery and
preclinical use.**
Hold out published nicotinic-receptor modulator data (e.g., ChEMBL on related
alpha subunits) for external validation of the enriched models before any
confirmatory use. Export versioned, AI-ready deliverables: matched-molecular-pair
transformation rules with activity deltas, per-chemotype structure-activity
summaries (including ascorbate sub-series SAR), versioned predictive models, and
the pre-registered analysis protocol — so that any future program can retrace
each SAR claim to raw data and code.

**Impact:** First experimentally enumerated ascorbate-derived alpha9alpha10 PAM
chemotype library; a reusable, honest, adaptive SAR-enrichment platform for
ion-channel targets; and an audit trail that converts machine-learning
predictions into chemically validated, reusable knowledge.