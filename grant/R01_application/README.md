# NIH R01 Grant Application Package
# alpha9alpha10 nAChR PAM chemotype library + closed-loop SAR enrichment

Submission-oriented set assembled from the verified pipeline artifacts in this
repository. All numbers, model metrics, and honesty caveats match the outputs
produced by the pipeline (see `scientific_paper.txt` and `reports/index.html`).

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
1. Zero executed rounds -> grant is framed as the pilot (Aim 2) with a
   comparator arm, not as completed enrichment.
2. n=30/7 actives -> models used only as "rankers to be validated"; external
   published-data validation is a planned Aim 1/3 task, not a completed claim.
3. Mechanism (TAF) outpaces evidence -> TAF retained as falsifiable hypotheses;
   specific aims are library+enrichment, mechanism is secondary.
4. No comparator arm -> two-arm design (model_ranked vs diversity_random) with
   pre-registration and enrichment-factor + FDR analysis (implemented).
5. Throughput mismatch -> staged design: 3 rounds x ~15 clean, drug-like
   compounds; candidate pool after filters is honestly 9/99 unique drug-like
   molecules (a real finding, see 02 section on library curation).
6. Endpoint too narrow -> add secondary selectivity/physchem gates + export of
   AI-ready enriched SAR for downstream preclinical use.
7. No statistical pre-specification -> pre-registration files per round with
   EF threshold, FDR method, calibration, confirmation criteria, go/no-go.

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