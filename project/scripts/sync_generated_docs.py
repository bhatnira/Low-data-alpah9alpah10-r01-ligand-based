#!/usr/bin/env python3
"""Post-generation doc sync for REINVENT claims.

Rewrites the honesty-sentinel prose (binary absent / NO molecules /
PENDING_REINVENT_BINARY) in the repo's narrative documents to the ACTUAL
generation results recorded in reinvent/reinvent_manifest.json +
reinvent/generated_molecules.csv.

Safety:
  * Refuses to touch anything unless generated_molecules.csv is non-empty.
  * Only replaces exact known sentinel phrases; every replacement keeps the
    claim "in silico / untested" and never fabricates activity.
  * --check never edits; exits 1 if stale REINVENT sentinels remain.

Usage:
  python3 scripts/sync_generated_docs.py            # apply
  python3 scripts/sync_generated_docs.py --check    # report only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent            # project/
REPO = ROOT.parent                                        # repo root

# REINVENT-only sentinel patterns; presence anywhere in DOCS => hard failure
RESIDUAL = re.compile(
    r"PENDING_REINVENT"
    r"|binary absent"
    r"|generates? NO molecules"
    r"|NO molecules exist"
    r"|no molecules generated yet"
    r"|produced 0 molecules"
    r"|-> 0 mols"
)

DOCS = [
    REPO / "grant/R01_application/01_specific_aims.md",
    REPO / "grant/R01_application/02_research_strategy.md",
    REPO / "nih_r01_application_v2.txt",
    REPO / "scientific_paper.txt",
    REPO / "project_architecture.txt",
]


def load_result() -> dict:
    mf = ROOT / "reinvent/reinvent_manifest.json"
    gf = ROOT / "reinvent/generated_molecules.csv"
    if not mf.exists() or not gf.exists() or gf.stat().st_size == 0:
        return {}
    manifest = json.loads(mf.read_text())
    status = manifest.get("run_status", "UNKNOWN")
    if not status.startswith(("RUN_", "PARTIAL_")):
        return {}
    n_rows = len(gf.read_text().splitlines()) - 1
    n_rows = len(gf.read_text().splitlines()) - 1
    return {
        "n": n_rows,
        "status": status,
        "device": manifest.get("device_resolved", "?"),
        "binary": manifest.get("reinvent_binary", ""),
        "version": manifest.get("reinvent_version", ""),
    }


def build_rules(f: dict) -> list:
    n = f["n"]
    st = f["status"]
    aims01 = (
        r"currently generates NO molecules \(binary absent; recorded\s+"
        r"READY_PENDING_REINVENT_BINARY, never fabricated\)"
    )
    aims02 = r"currently NO molecules exist because the binary is absent"
    v2_gen = (
        r"REINVENT4 generates NO molecules\s+\(binary absent; recorded\s+"
        r"READY_PENDING_REINVENT_BINARY, never fabricated\)"
    )
    v2_yet = r"REINVENT4: no molecules generated yet \(binary absent\) - reported with provenance, not fabricated\."
    v2_562 = (
        r"REINVENT4 Mol2Mol generation \(stereo-aware prior\): currently NO molecules\s+"
        r"exist because the binary is absent"
    )
    sci_cfg = (
        r"configs written and checksum-validated; binary NOT AVAILABLE in\s+"
        r"repository -> generation pending; NO molecules fabricated\."
    )
    sci_l527 = (
        r"5\. REINVENT binary absent\. Generative novelty metrics are computed only when\s+"
        r"molecules exist; none fabricated\."
    )
    sci_l636 = (
        r"5\. REINVENT binary absent\s+-> generator registry scans for newly produced\s+"
        r"molecules each cycle; nothing fabricated"
    )
    arch_tree = r"reinvent/\s+REINVENT4 configs \+ generation plan \(binary absent -> 0 mols\)"
    arch_flow = r"\| 9-11\. reinvent\| \(binary absent -> NO molecules generated; provenance kept\)"
    arch_honest = r"REINVENT has produced 0 molecules \(binary absent\) - reported, not faked\."

    return [
        (aims01, f"currently generated {n} molecules (all in silico, untested; {st})"),
        (aims02, f"currently generated {n} molecules (all in silico, untested; {st})"),
        (v2_gen, f"REINVENT4 generated {n} molecules (all in silico, untested; {st}, never fabricated)"),
        (v2_yet, f"REINVENT4: {n} molecules generated (in silico, untested) - reported with provenance, not fabricated."),
        (v2_562, f"REINVENT4 Mol2Mol generation (stereo-aware prior): currently {n} molecules generated (all in silico, untested; {st})"),
        (sci_cfg, f"configs written and checksum-validated; binary present -> generation run ({st}); {n} molecules collected; no molecules fabricated."),
        (sci_l527, f"5. REINVENT binary present ({n} molecules generated). Generative novelty metrics are computed only when molecules exist; none fabricated."),
        (sci_l636, f"5. REINVENT binary present ({n} molecules) -> generator registry scans for newly produced molecules each cycle; nothing fabricated"),
        (arch_tree, f"reinvent/        REINVENT4 configs + generation plan ({n} mols generated, untested)"),
        (arch_flow, f"| 9-11. reinvent| ({n} molecules generated, untested; provenance kept)"),
        (arch_honest, f"REINVENT has produced {n} molecules (in silico, untested) - reported with provenance, not faked."),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, never edit")
    args = ap.parse_args()

    result = load_result()
    if not result:
        print("sync_generated_docs: no real generation results "
              "(generated_molecules.csv empty/missing) - refusing to touch docs.")
        return 0 if not args.check else 3

    print(f"facts: n={result['n']} status={result['status']} "
          f"device={result['device']} version={result['version']}")
    rules = build_rules(result)
    ok = True
    for doc in DOCS:
        if not doc.exists():
            print(f"  - skip (missing): {doc.relative_to(REPO)}")
            continue
        text = doc.read_text()
        changed = 0
        for pat, sub in rules:
            new_text, nsub = re.subn(pat, sub, text, flags=re.DOTALL)
            if nsub and new_text != text:
                changed += nsub
                text = new_text
                if not args.check:
                    doc.write_text(text)
        if changed:
            print(f"  - {doc.relative_to(REPO)}: {changed} replacement(s) applied "
                  f"{'(dry-run)' if args.check else ''}")
        else:
            print(f"  - {doc.relative_to(REPO)}: no matching sentinels")

    # hard gate: no stale REINVENT-only sentinels may remain anywhere in scope
    stale = []
    for doc in DOCS:
        if not doc.exists():
            continue
        for i, line in enumerate(doc.read_text().splitlines(), 1):
            if RESIDUAL.search(line):
                stale.append(f"  {doc.relative_to(REPO)}:{i}: {line.strip()}")
    if stale:
        print("STALE REINVENT sentinels (fix manually or rerun after a run):")
        print("\n".join(stale))
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())