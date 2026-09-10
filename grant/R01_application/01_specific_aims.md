# SPECIFIC AIMS (1 page)

**Title:** Development of High-Potency, Selective Ascorbate-Derived
alpha9alpha10 Nicotinic Acetylcholine Receptor (nAChR) Positive Allosteric
Modulators (PAMs) through an Integrated Medicinal Chemistry and AI-Guided
Ligand-Based Approach

alpha9alpha10 nAChRs are implicated in chronic pain, tinnitus, and cochlear
damage, yet no PAM chemotype has been translated, and publicly available
modulator chemistry fails to transfer to this rare subunit combination. Our
computational workflow (TAF = transferable activity functions) has curated 30
validated compounds (7 PAM actives), all sharing the L-ascorbic acid
(enediol-gamma-lactone) pharmacophore — the only validated PAM chemotype for
this target. We generated a 15-compound prospective candidate portfolio
(9 chemistry-validated analogs plus 6 negative controls). The
gap is that (i) modest structural modifications of ascorbic acid dramatically
enhance alpha9alpha10 PAM potency (ID12 3-O-substituted-5,6-acetonide,
0.198 uM, ~9,100x below the 1797 uM reference; ID25 6-bromo-6-deoxy,
2.63 uM) but the structural features that govern potentiation are not yet
systematically mapped, and (ii) no experimental round has demonstrated that a
predictive model can guide those edits. Our long-term goal is a
medicinal-chemistry-driven analog series of high-potency, selective
alpha9alpha10 PAMs, supported by an AI-guided platform that prioritizes and
validates candidates for any future drug-discovery or preclinical program.

Route-prediction decision support (AIZynthFinder retrosynthesis) is live:
1/8 round-1 candidates carries a PREDICTED_ROUTE/UNVERIFIED plan (1-step,
route product verified against the target) plus a per-building-block stock
screen (18/38 blocks IN_STOCK in ZINC) for route-planning review, so
"synthesizable" is reviewed against reproducible evidence, and explicit
NO_ROUTE_FOUND (7/8 targets) is reported rather than fabricated.

**Aim 1 (single integrated aim). Develop high potency and selective
ascorbate-derived alpha9alpha10 positive allosteric modulators through an
integrated medicinal chemistry and AI-guided ligand-based approach.**

Our preliminary studies demonstrate that modest structural modifications of
ascorbic acid can dramatically enhance alpha9alpha10 PAM potency. We will
define the structural requirements governing alpha9alpha10 PAM activity
through systematic SAR analysis and use computational and AI-guided approaches
to capture and harness these relationships to guide compound design and
prioritization in an iterative closed-loop medicinal chemistry workflow,
followed by synthesis and evaluation using two-electrode voltage-clamp
electrophysiology. This aim is innovative in its dual focus: (1) systematic SAR of ascorbate to
define structural requirements for alpha9alpha10 PAM activity, and (2)
strategic design of transporter-compatible analogs to facilitate targeted
cochlear delivery; advanced synthetic methodologies and mechanistic design
enhance the potential for developing dual-acting compounds. Since we have
already succeeded in identifying a significantly more potent analog of
ascorbic acid (ID12, 0.198 uM), the success of this aim appears likely — and
is tested empirically in pre-registered rounds.

Four coupled activities deliver this aim in parallel:

- **1a. Medicinal chemistry + SAR-guided analog design.** Systematically edit
  the ascorbate envelope around the position-resolved SAR: 3-O-alkoxy +
  5,6-acetonide is the high-potency arm (ID12, 0.198 uM), 6-bromo-6-deoxy
  (ID25, 2.63 uM) second, 2-O-alkoxy / 6-O-benzyl weak actives, 2,3-bis-O-
  alkylation and non-ascorbate inactive. The synthetic program is organized
  into five integrated studies, each probing a defined structural feature:
  Study 1.1 (role of the C-2 hydroxyl — 2-deoxy/2-halo/2-O-acyl/2-O-alkyl
  series), Study 1.2 (conformationally rigid bicyclic ascorbate analogs),
  Study 1.3 (C-3-O-heterocyclic analogs along the highest-potency vector),
  Study 1.4 (C-5/C-6 hydroxyl role — deoxygenation/olefination, ~10 analogs),
  and Study 1.5 (deoxy-ascorbate mimics and azalactams via Dutta-lactone
  bicyclics, ~8/category). 56 ascorbate-restricted MMP transforms guide
  enumeration. All analogs pass pre-registered PAINS/BRENK/Lipinski gates and
  are reported with provenance, not fabricated.
- **1b. Computational design support (supplements 1a, not a separate
  deliverable).** REINVENT4 Mol2Mol (stereo-aware prior) with an openly
  documented rank score (predicted activity x information-gain boost,
  scaffold-capped) is the planned in-loop generator; consensus explanations
  flag which proposed edit drives potency. Responsibly scoped: REINVENT4
  currently generates NO molecules (binary absent; recorded
  READY_PENDING_REINVENT_BINARY, never fabricated) — the current prospective
  set therefore comes from the traditional medicinal-chemistry baseline
  (97 enumerated / 86 chemistry-invalid / 11 valid; 9 carried into the
  candidate set plus 6 negative controls). When installed, generator output
  passes the same pre-registered filters and is reported as prospective;
  computational proposals feed, they do not replace, the medicinal-chemistry
  analog plan.
- **1c. Iterative electrophysiological evaluation.** Each round tests drug-like
  analogs (round 1 = 6, currently BLOCKED pending med-chem sign-off) split into
  a model-ranked arm and an equal-size diversity-randomized control arm
  (matched on chemotype tier), blinded to the team. TEVC potentiation assays on
  alpha9alpha10-expressing
  Xenopus oocytes (co-PI laboratory), each compound >=3 oocytes, blind operator
  to arm. Primary endpoint: potentiation>0 AND potency>0, confirmed by
  independent dose-response. Oversight gate: medicinal-chemistry sign-off on
  synthesizability (phase 4c route plans), stereo feasibility, assay
  suitability, TAF consistency, negative-control balance; no experiment
  proceeds before approval.
- **1d. Predictive model construction + iterative refit.** After each round,
  append results (never overwrite), refit the predictive model with the LAST
  round held out, update TAF evidence and power from pooled variability (no
  post-hoc re-powering). Pre-registered analyses: enrichment factor
  (threshold 2.0), BH-FDR (alpha=0.10), precision@k, arm-stratified hit rates,
  go/no-go (continue iff EF>=2.0 and >=1 clean novel chemotype confirmed; max
  3 rounds). Deliver >=3 enriched-SAR releases and a validated predictive
  model of alpha9alpha10 PAM activity that guides all subsequent analog design.

Selectivity and synthesizability are enforced inside the loop, not as separate
aims: before plate locking, each analog's frozen-model score across a curated,
isolated 70-compound non-alpha9alpha10 panel (phase 4b; 0 exact-SMILES / 0
scaffold overlap; baseline 0/70 >= 0.5, mean 0.215 +/- 0.045) and its
route-prediction plan + building-block stock status (phase 4c) are re-checked
and reported in the pre-registration file.

**Impact:** First experimentally enumerated, high-potency and selective
ascorbate-derived alpha9alpha10 PAM series; a validated predictive model of
alpha9alpha10 PAM activity that supports (not replaces) medicinal-chemistry
decisions; a selectivity benchmark making off-target cross-subtype activity an
explicit measured design axis; and an audit trail that converts
machine-learning predictions into chemically validated, reusable knowledge.