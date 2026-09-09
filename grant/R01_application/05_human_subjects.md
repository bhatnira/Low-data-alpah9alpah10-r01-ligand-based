# HUMAN SUBJECTS / VERTEBRATE ANIMALS / RESOURCE SHARING (checklist)

## Human subjects: NOT APPLICABLE
This project involves no human subjects, human tissues, or identifiable
personal data. Assays use heterologously expressed ion channels in Xenopus
oocytes. The IRB exemption box is checked on the SF424.

## Vertebrate animals
See `06_vertebrate_animals.md`. Briefly: oocytes are purchased from a commercial
supplier as isolated cells; no live vertebrate animal work is performed at the
awardee institution. If the institution treats oocyte procurement as covered
animal work, the IACUC protocol number must be provided at submission.

## Resource Sharing Plan
- **Model/data sharing:** all versioned predictive models
  (`models/classical/lbm_rf_fp_desc_rN.joblib`), pooled datasets, and
  pre-registration files will be deposited at a DOI-bearing public archive
  (e.g., Figshare/Zenodo) with the code release below; no confidential or
  proprietary data exist at submission.
- **Software:** source code (RDKit, scikit-learn, scipy; no closed binaries) is
  released under a permissive license with pinned dependency versions, seeds,
  and a containerized environment so claims are fully reproducible.
- **Reagents:** verified compounds purchased from CROs are inventoried with
  vendor lot + purity; cells/oocytes commercially sourced.

## Data Management and Sharing Plan (NIH Policy)
1. Data types: RAW assay traces/NEX files, canonicalized structures,
   pooled datasets, model artifacts, reports, pre-registration JSONs,
   provenance manifests.
2. Standards/metadata: every artifact carries a provenance manifest
   (software versions, seeds, config hash, input hashes, git commit); chemical
   structures in canonical SMILES with stereochemical handling documented.
3. Preservation: append-only raw-data archive, versioned, hash-checked; minimum
   of the grant period + 5 years.
4. Access/share: embargo-free release at each round milestone (per NIH's
   Data Management policy for no-clinical-data awards); no proprietary or
   privacy barriers.
5. Oversight/compliance: governance per institutional policy; integrity checks
   (leakage audit, overclaim scan) run automatically at every pipeline pass and
   their results published with the data.

## 18-Question Final Audit (evidence, not assertion)
The pipeline's automated 18-question "before you finish" audit and leakage
audit pass at every full run; results are in `reports/index.html` (Appendix B,
Appendix A). Citation of these audits in the Research Strategy is backed by the
actual run manifest.