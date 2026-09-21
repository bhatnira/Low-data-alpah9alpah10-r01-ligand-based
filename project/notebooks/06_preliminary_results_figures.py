"""
Generate two figures for R01 Preliminary Results - Aim 1.
Figure 1: Scaffold frequency + TAF pharmacophore map
Figure 2: PCA chemical space + hierarchical selection funnel
"""

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---------- load data ----------
train = pd.read_csv(ROOT / "data" / "processed" / "data_cleaned.csv")
gen = pd.read_csv(ROOT / "generated" / "generated_predictions.csv")
portfolio = pd.read_csv(ROOT / "portfolio" / "final_candidate_portfolio.csv")
scaffold = pd.read_csv(ROOT / "sar" / "scaffolds" / "scaffold_analysis.csv")
taf = pd.read_csv(ROOT / "taf" / "discovery" / "taf_blueprint.csv")

# ---------- compute Morgan fingerprints ----------
def smiles_to_fp(smi, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

print("Computing fingerprints for training set...")
train_fps = []
train_idx = []
for i, smi in enumerate(train["canonical_smiles"]):
    fp = smiles_to_fp(smi)
    if fp is not None:
        train_fps.append(fp)
        train_idx.append(i)
train_fps = np.array(train_fps)
train_labels = train.loc[train_idx, "is_active"].values

print("Computing fingerprints for generated library (sample 5000)...")
rng = np.random.RandomState(42)
sample_n = min(5000, len(gen))
sample_idx = rng.choice(len(gen), sample_n, replace=False)
gen_fps = []
gen_sample_idx = []
for i in sample_idx:
    fp = smiles_to_fp(gen.iloc[i]["SMILES"])
    if fp is not None:
        gen_fps.append(fp)
        gen_sample_idx.append(i)
gen_fps = np.array(gen_fps)

# ---------- PCA on combined fingerprints ----------
print("Running PCA...")
all_fps = np.vstack([train_fps, gen_fps])
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(all_fps)
train_coords = coords[:len(train_fps)]
gen_coords = coords[len(train_fps):]

# ============================================================
# Figure 1: Scaffold frequency + TAF map (2 panels)
# ============================================================
print("Generating Figure 1...")
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# --- Panel A: Scaffold frequency ---
ax = axes[0]
scaffold_sorted = scaffold.sort_values("count", ascending=True)
colors = ["#d62728" if n_a > 0 else "#4393c3" for n_a in scaffold_sorted["n_active"]]
bars = ax.barh(range(len(scaffold_sorted)), scaffold_sorted["count"], color=colors,
               edgecolor="white", linewidth=0.8, alpha=0.9)
ax.set_yticks(range(len(scaffold_sorted)))
ax.set_yticklabels(scaffold_sorted["scaffold"], fontsize=9, fontfamily="monospace")
ax.set_xlabel("Number of compounds", fontsize=10)
ax.set_title("A. Bemis-Murcko Scaffold Frequency\n(30-compound training set)", fontsize=11, fontweight="bold")

# add count labels
for i, (count, n_a) in enumerate(zip(scaffold_sorted["count"], scaffold_sorted["n_active"])):
    ax.text(count + 0.1, i, f"{count} ({n_a} active)", va="center", fontsize=8.5)

legend_elements = [
    mpatches.Patch(facecolor="#d62728", label="Contains active compounds"),
    mpatches.Patch(facecolor="#4393c3", label="Inactive only"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel B: TAF pharmacophore schematic ---
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_aspect("equal")
ax2.axis("off")
ax2.set_title("B. Ascorbate Pharmacophore (TAF Map)", fontsize=11, fontweight="bold")

# draw lactone core (pentagon)
lactone_pts = np.array([
    [4.0, 4.0], [5.5, 3.0], [6.5, 4.2], [5.8, 5.5], [4.5, 5.2]
])
lactone_pts = np.vstack([lactone_pts, lactone_pts[0]])
ax2.plot(lactone_pts[:, 0], lactone_pts[:, 1], "k-", linewidth=2)
ax2.fill(lactone_pts[:, 0], lactone_pts[:, 1], alpha=0.15, color="#d62728")
ax2.text(5.2, 4.2, "γ-lactone\ncore", ha="center", va="center", fontsize=9, fontweight="bold")

# TAF-1: lactone ring
ax2.annotate("TAF-1\nLactone ring\n(H-bond acceptor)\n[REQUIRED]",
             xy=(5.0, 3.0), xytext=(1.5, 1.0),
             fontsize=8, ha="center", color="#d62728", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffe0e0", edgecolor="#d62728"))

# TAF-2: hydroxyl donor
ax2.plot([6.5, 7.8], [4.2, 3.0], "b-", linewidth=1.5)
ax2.plot(7.8, 3.0, "bo", markersize=8)
ax2.text(8.0, 2.7, "OH", fontsize=9, fontweight="bold", color="blue")
ax2.annotate("TAF-2\nHydroxyl donor\n(~109° vector)\n[PREFERRED]",
             xy=(7.8, 3.0), xytext=(8.5, 1.0),
             fontsize=8, ha="center", color="blue", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#e0e0ff", edgecolor="blue"))

# TAF-3: EWG 3-5A from core
ax2.annotate("TAF-3\nEWG 3–5 Å\nfrom core\n[REQUIRED]",
             xy=(4.0, 5.2), xytext=(1.5, 7.5),
             fontsize=8, ha="center", color="#2ca02c", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#e0ffe0", edgecolor="#2ca02c"))

# TAF-4: stereochemistry
ax2.annotate("TAF-4\nDefined\nstereochemistry\n[REQUIRED]",
             xy=(5.8, 5.5), xytext=(8.0, 7.5),
             fontsize=8, ha="center", color="#9467bd", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.5),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0e0ff", edgecolor="#9467bd"))

# stereo centers marked with *
ax2.text(5.8, 5.8, "*", fontsize=14, ha="center", color="#9467bd", fontweight="bold")

fig.tight_layout(w_pad=3)
fig.savefig(FIG / "prelim_fig1_scaffold_taf.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "prelim_fig1_scaffold_taf.pdf", bbox_inches="tight")
plt.close(fig)
print(f"  Saved to {FIG / 'prelim_fig1_scaffold_taf.png'}")

# ============================================================
# Figure 2: PCA space + Selection funnel (2 panels)
# ============================================================
print("Generating Figure 2...")
fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.1, 1]})

# --- Panel A: PCA chemical space ---
ax = axes[0]
ax.scatter(gen_coords[:, 0], gen_coords[:, 1],
           c="#d0d0d0", s=8, alpha=0.4, zorder=1, label="Generated (n=31,474)")
inactive_mask = train_labels == 0
ax.scatter(train_coords[inactive_mask, 0], train_coords[inactive_mask, 1],
           c="#4393c3", s=80, edgecolors="white", linewidths=0.8,
           zorder=3, marker="o", label="Training – inactive (n=23)")
active_mask = train_labels == 1
ax.scatter(train_coords[active_mask, 0], train_coords[active_mask, 1],
           c="#d62728", s=120, edgecolors="white", linewidths=0.8,
           zorder=4, marker="*", label="Training – active (n=7)")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)", fontsize=10)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)", fontsize=10)
ax.set_title("A. Chemical Space Expansion\naround Ascorbate Scaffold", fontsize=11, fontweight="bold")
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel B: Hierarchical funnel ---
ax2 = axes[1]
levels = [
    ("Training\nSet", 30, "#4393c3"),
    ("Generated\nLibrary", 31474, "#92c5de"),
    ("IG-Gate\nPool", 2788, "#f4a582"),
    ("Candidate\nPortfolio", 100, "#d6604d"),
    ("Tier 2\nMechanistic", 17, "#b2182b"),
    ("Route-\nReady", 11, "#67001f"),
]

y_positions = list(range(len(levels)))[::-1]
bar_heights = [np.log10(max(v, 1)) for _, v, _ in levels]
max_h = max(bar_heights)

for i, (label, count, color) in enumerate(levels):
    y = y_positions[i]
    w = bar_heights[i] / max_h
    ax2.barh(y, w, height=0.7, color=color, edgecolor="white", linewidth=1.2, alpha=0.9)
    ax2.text(w + 0.02, y, f"{count:,}", va="center", ha="left", fontsize=10, fontweight="bold")

ax2.set_yticks(y_positions)
ax2.set_yticklabels([l for l, _, _ in levels], fontsize=9)
ax2.set_xlabel("Relative scale (log₁₀)", fontsize=10)
ax2.set_title("B. Hierarchical Candidate Selection", fontsize=11, fontweight="bold")
ax2.set_xlim(0, 1.35)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout(w_pad=3)
fig.savefig(FIG / "prelim_fig2_space_funnel.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "prelim_fig2_space_funnel.pdf", bbox_inches="tight")
plt.close(fig)
print(f"  Saved to {FIG / 'prelim_fig2_space_funnel.png'}")

print("Done.")
