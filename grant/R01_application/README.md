# NIH R01 Grant Application Package (v2)
# alpha9alpha10 nAChR PAM chemotype library + closed-loop SAR enrichment

Submission-oriented set assembled from the verified pipeline artifacts in this
repository. All numbers, model metrics, and honesty caveats match the outputs
produced by the pipeline (see `scientific_paper.txt` and `reports/index.html`).
A single-file consolidated submission draft is available at
`../../nih_r01_application_v2.txt` (v1 preserved as `../../nih_r01_application.txt`).

## Package contents
| File | Corresponding NIH form / section |
|---|---|
| `00_project_summary.md` | SF424 Project Summary/Abstract (performance site, budget, etc.) |
| `01_specific_aims.md` | Specific Aims (1 page) |
| `02_research_strategy.md` | Research Strategy (Significance / Innovation / Approach) |
| `03_budget_justification.md` | Budget + justification (modular) |
| `04_biosketch_template.md` | NIH Biosketch template (PI + key personnel placeholders) |
| `05_human_subjects.md` | Human Subjects / Vertebrate Animals / Resource Sharing (exempt) |
| `06_vertebrate_animals.md` | Vertebrate Animals section (Xenopus oocytes) |
| `07_letters_of_support_template.md` | Letters of Support templates |
| `08_reviewer_responses.md` | Response to the seven review critiques + where each is handled |

## How the earlier review critiques are addressed (summary)
1. Zero executed rounds -> grant is framed as the pilot (Aim 1 / iterative
   rounds) with a comparator arm, not as completed enrichment.
2. n=30/7 actives -> models used only as "rankers to be validated"; external
   published-data validation is a planned downstream task, not a completed
   claim.
3. Mechanism (TAF) outpaces evidence -> TAF retained as falsifiable hypotheses;
   the single integrated aim is analog optimization + model building, mechanism
   is secondary.
4. No comparator arm -> two-arm design (model_ranked vs diversity_random) with
   pre-registration and enrichment-factor + FDR analysis (implemented).
5. Throughput mismatch -> staged design: 3 rounds x ~15 clean, drug-like
   compounds; candidate pool after filters is honestly 6/15 unique drug-like
   molecules (a real finding, see 02 section on library curation).
6. Endpoint too narrow -> add secondary selectivity/physchem gates + a curated
   non-alpha9alpha10 selectivity panel (phase 4b, 70 modulators, isolation
   contract enforced), plus route-prediction decision support (phase 4c)
   making synthesizability reproducible in-loop.
7. No statistical pre-specification -> pre-registration files per round with
   EF threshold, FDR method, calibration, confirmation criteria, go/no-go.

## v2 changes (vs v1 txt)
- Reframed to a SINGLE integrated Aim 1: high-potency, selective
  ascorbate-derived alpha9alpha10 PAMs via a medicinal-chemistry-driven analog
  program (studies 1.1-1.5) with computational/AI guidance as a supplement -
  iterative electrophysiological evaluation and a predictive model that
  supports (not replaces) medicinal-chemistry decisions (matches
  `00`/`01`/`02`).
- Term shortened to 3 years (09/01/2027 - 08/31/2030).
- Kept the phase-4b selectivity panel (74 raw entries -> 70 curated, isolation
  contract) and added the phase-4c retrosynthetic route-prediction decision
  support
  (AIZynthFinder, public USPTO/ZINC models; PREDICTED_ROUTE/UNVERIFIED plans +
  building-block stock screen feeding the oversight gate; explicit
  NO_ROUTE_FOUND, never fabricated). These are in-loop design axes (Innovation
  #5/#6, 3.3.4/3.3.5), not separate aims.
- Pipeline upgraded from 17 to 19 phases with the new `selectivity` phase
  (Phase 4b, after model training) and the new `synth_route` phase
  (Phase 4c, after prospective/lock, before loop).

## Readiness checklist
- [ ] Replace `[PI NAME]` and other bracketed placeholders (04).
- [ ] Attach real budget/joint support; adjust oocyte/assay costs to institution.
- [ ] Add coversheet (SF424), PHS 398 forms, and submit via ASSIST/eRA Commons.
- [ ] If a wet-lab collaborator exists, promote to co-PI/co-I and add biosketch.
- [ ] Run `python workflows/run_all.py` in `/project` to regenerate artifacts
      with the correct as-of date before submission.

## Honesty statement
This package contains NO fabricated experimental results, personnel, letters,
or funding. Everything labeled "PENDING", "PROSPECTIVE", or "planned" is a plan
whose machinery exists and runs in this repository. The Experimental section of
the Research Strategy is forward-looking and must be reviewed by a real wet-lab
PI before submission.