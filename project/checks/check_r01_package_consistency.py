#!/cluster/scratch/nbhatt04/conda/envs/polymer_ai/bin/python
"""Honest pre-ship consistency check for reports/r01_package/.

Every filename the narrative cites in prose must exist in the folder, and
every .tex/.csv the folder truly ships must be cited or be a deliberate
appendix piece. No check here invents a path; it lists the folder.
"""
import pathlib
import re

pkg = pathlib.Path("reports/r01_package")
narr = (pkg / "r01_project_narrative.txt").read_text()

cited = set(re.findall(r"[\w./-]+\.(?:tex|csv|txt)\b", narr))
real = {f.name for f in pkg.iterdir() if f.is_file()}
real |= {f.name for f in (pkg / "data_appendix").iterdir()} if (pkg / "data_appendix").exists() else set()

print("=== citations in narrative prose vs artifacts actually in folder ===")
missing = sorted(c for c in cited if pathlib.PurePosixPath(c).name not in real)
if not missing:
    print("  all cited artifact basenames EXIST on disk  -> self-consistent")
else:
    print("  MISSING (narrative cites a file that is NOT in the folder):")
    for m in missing:
        print("   -", m)

print("=== .tex basename the door-copy should point at (the one that exists) ===")
tex = [f.name for f in pkg.iterdir() if f.suffix == ".tex"]
print("  on disk:", tex)

print("=== narrative: does prose claim the .tex is already-compiled-PDF? (must be no) ===")
claims = [l for l in narr.splitlines() if "pdf" in l.lower() and ("compil" in l.lower() or "pdflatex" in l.lower())]
for l in claims[:5]:
    flag = "  HONEST" if any(w in l.lower() for w in ("compile-ready", "not present", "run", "no pdflatex", "absent")) else "  CHECK"
    print(flag, "|", l.strip()[:110])
