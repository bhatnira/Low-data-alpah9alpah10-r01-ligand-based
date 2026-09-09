#!/usr/bin/env python3
"""Steps 2-4: Activity Representation, Exploratory Cheminformatics, Common-Core Analysis."""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Descriptors, AllChem, rdMolDescriptors, rdFingerprintGenerator,
    Draw, rdmolops
)
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import networkx as nx
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────────────────────────
PROJECT = '/Users/nb/Documents/ligand-based-modeling/project'
QC_CSV = os.path.join(PROJECT, 'data/qc/data_qc.csv')
FIG_DIR = os.path.join(PROJECT, 'figures')
OUT_DIR = os.path.join(PROJECT, 'analysis')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'exploratory'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'SAR'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'MMP'), exist_ok=True)

df = pd.read_csv(QC_CSV)
print(f"Loaded {len(df)} compounds from QC data")

# Parse mols
df['mol'] = df['canonical_smiles'].apply(Chem.MolFromSmiles)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: ACTIVITY REPRESENTATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: ACTIVITY REPRESENTATION")
print("=" * 70)

act_col = 'Activity (uM)'
pot_col = '%Potentiation'

# Potency transformation: pActivity = -log10(uM)
# Only for non-zero values
df['pActivity'] = np.where(
    df[act_col] > 0,
    -np.log10(df[act_col] / 1e6),  # Convert uM to M, then -log10
    np.nan
)
# Alternative: -log10(uM directly) for simpler interpretation
df['pPotency'] = np.where(
    df[act_col] > 0,
    -np.log10(df[act_col]),
    np.nan
)

print("Activity representations created:")
print(f"  pPotency (-log10[uM]): range = [{df['pPotency'].min():.2f}, {df['pPotency'].max():.2f}]")
print(f"  Active compounds with pPotency: {df['pPotency'].notna().sum()}")

# Potency categories
df['potency_category'] = 'inactive'
df.loc[(df[act_col] > 0) & (df[act_col] <= 10), 'potency_category'] = 'high_potency'
df.loc[(df[act_col] > 10) & (df[act_col] <= 1000), 'potency_category'] = 'moderate_potency'
df.loc[df[act_col] > 1000, 'potency_category'] = 'low_potency'

print("\nPotency categories:")
print(df['potency_category'].value_counts().to_string())

# Normalize potentiation (0-1)
max_pot = df[pot_col].max()
df['norm_potentiation'] = df[pot_col] / max_pot if max_pot > 0 else 0

# Joint activity metric
df['activity_score'] = 0.0
active_mask = df['is_active'] == 1
if active_mask.sum() > 0:
    # Normalize potency score (lower uM = higher score)
    act_uM = df.loc[active_mask, act_col]
    potency_score = 1 - (act_uM - act_uM.min()) / (act_uM.max() - act_uM.min() + 1e-10)
    pot_score = df.loc[active_mask, pot_col] / df.loc[active_mask, pot_col].max()
    df.loc[active_mask, 'activity_score'] = 0.7 * potency_score + 0.3 * pot_score

print(f"\nActivity score range: [{df['activity_score'].min():.3f}, {df['activity_score'].max():.3f}]")

# Save activity table
activity_cols = ['Identifier', 'canonical_smiles', act_col, pot_col, 
                 'is_active', 'potency_rank', 'pPotency', 'potency_category',
                 'norm_potentiation', 'activity_score']
df[activity_cols].to_csv(os.path.join(OUT_DIR, 'exploratory/activity_representation.csv'), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: EXPLORATORY CHEMoinFORMATICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: EXPLORATORY CHEMoinFORMATICS")
print("=" * 70)

# ── 2D FINGERPRINTS ─────────────────────────────────────────────────────────
print("\n--- Generating Fingerprints ---")

def get_morgan_fp(mol, radius=2, nbits=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(nbits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def get_maccs_fp(mol):
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    arr = np.zeros(167, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def get_rdkit_fp(mol, nbits=2048):
    fp = AllChem.RDKFingerprint(mol, fpSize=nbits)
    arr = np.zeros(nbits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# Morgan fingerprints (radius 1,2,3)
for r in [1, 2, 3]:
    col_name = f'morgan_r{r}'
    df[col_name] = df['mol'].apply(lambda m: get_morgan_fp(m, radius=r) if m else np.zeros(2048))
    print(f"  Morgan radius {r}: {df[col_name].apply(lambda x: x.sum()).mean():.1f} bits set on average")

# MACCS keys
df['maccs'] = df['mol'].apply(lambda m: get_maccs_fp(m) if m else np.zeros(167))
print(f"  MACCS keys: {df['maccs'].apply(lambda x: x.sum()).mean():.1f} bits set on average")

# RDKit topological FP
df['rdkit_fp'] = df['mol'].apply(lambda m: get_rdkit_fp(m) if m else np.zeros(2048))
print(f"  RDKit FP: {df['rdkit_fp'].apply(lambda x: x.sum()).mean():.1f} bits set on average")

# ── MOLECULAR SIMILARITY ────────────────────────────────────────────────────
print("\n--- Computing Pairwise Similarity ---")

n = len(df)
sim_matrix_morgan = np.zeros((n, n))
sim_matrix_maccs = np.zeros((n, n))
sim_matrix_rdkit = np.zeros((n, n))

for i in range(n):
    for j in range(i, n):
        fp_i_m = df.iloc[i]['mol']
        fp_j_m = df.iloc[j]['mol']
        if fp_i_m is None or fp_j_m is None:
            continue
        # Morgan
        fp_i = get_morgan_fp(fp_i_m, radius=2)
        fp_j = get_morgan_fp(fp_j_m, radius=2)
        tan = np.sum(fp_i & fp_j) / np.sum(fp_i | fp_j) if np.sum(fp_i | fp_j) > 0 else 0
        sim_matrix_morgan[i, j] = sim_matrix_morgan[j, i] = tan
        # MACCS
        fp_i = get_maccs_fp(fp_i_m)
        fp_j = get_maccs_fp(fp_j_m)
        tan = np.sum(fp_i & fp_j) / np.sum(fp_i | fp_j) if np.sum(fp_i | fp_j) > 0 else 0
        sim_matrix_maccs[i, j] = sim_matrix_maccs[j, i] = tan
        # RDKit
        fp_i = get_rdkit_fp(fp_i_m)
        fp_j = get_rdkit_fp(fp_j_m)
        tan = np.sum(fp_i & fp_j) / np.sum(fp_i | fp_j) if np.sum(fp_i | fp_j) > 0 else 0
        sim_matrix_rdkit[i, j] = sim_matrix_rdkit[j, i] = tan

print(f"  Morgan similarity range: [{sim_matrix_morgan[np.triu_indices(n, k=1)].min():.3f}, {sim_matrix_morgan[np.triu_indices(n, k=1)].max():.3f}]")
print(f"  Morgan similarity mean: {sim_matrix_morgan[np.triu_indices(n, k=1)].mean():.3f}")

# Save similarity matrices
np.save(os.path.join(OUT_DIR, 'exploratory/similarity_morgan.npy'), sim_matrix_morgan)
np.save(os.path.join(OUT_DIR, 'exploratory/similarity_maccs.npy'), sim_matrix_maccs)
np.save(os.path.join(OUT_DIR, 'exploratory/similarity_rdkit.npy'), sim_matrix_rdkit)

# ── PCA ─────────────────────────────────────────────────────────────────────
print("\n--- PCA Analysis ---")

# Stack Morgan radius 2 fingerprints
fps = np.vstack(df['morgan_r2'].values)
scaler = StandardScaler()
fps_scaled = scaler.fit_transform(fps)

pca = PCA(n_components=min(10, n))
pca_result = pca.fit_transform(fps_scaled)
df['pca_1'] = pca_result[:, 0]
df['pca_2'] = pca_result[:, 1]
if pca_result.shape[1] > 2:
    df['pca_3'] = pca_result[:, 2]

print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
print(f"  Cumulative variance (2 PCs): {sum(pca.explained_variance_ratio_[:2]):.3f}")

# ── t-SNE ───────────────────────────────────────────────────────────────────
print("\n--- t-SNE Analysis ---")
tsne = TSNE(n_components=2, perplexity=min(8, n-1), random_state=42, max_iter=2000)
tsne_result = tsne.fit_transform(fps_scaled)
df['tsne_1'] = tsne_result[:, 0]
df['tsne_2'] = tsne_result[:, 1]

# ── HIERARCHICAL CLUSTERING ─────────────────────────────────────────────────
print("\n--- Hierarchical Clustering ---")
dist_matrix = 1 - sim_matrix_morgan
np.fill_diagonal(dist_matrix, 0)
condensed_dist = squareform(dist_matrix)
Z = linkage(condensed_dist, method='ward')

# Cut into clusters
n_clusters = 4
clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
df['cluster'] = clusters
print(f"  Number of clusters: {n_clusters}")
for c in range(1, n_clusters + 1):
    members = df[df['cluster'] == c]
    print(f"  Cluster {c}: {len(members)} compounds, IDs: {members['Identifier'].tolist()}")

# ── FIGURES ─────────────────────────────────────────────────────────────────
print("\n--- Generating Figures ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. PCA colored by activity
ax = axes[0, 0]
colors = df['potency_category'].map({
    'high_potency': 'red', 'moderate_potency': 'orange',
    'low_potency': 'yellow', 'inactive': 'gray'
})
ax.scatter(df['pca_1'], df['pca_2'], c=colors, s=80, edgecolors='black', linewidth=0.5)
for _, r in df.iterrows():
    ax.annotate(str(int(r['Identifier'])), (r['pca_1'], r['pca_2']), fontsize=7, ha='center', va='bottom')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA - Chemical Space')
ax.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='High potency'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Moderate'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=8, label='Low potency'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Inactive')
], loc='best', fontsize=8)

# 2. t-SNE colored by activity
ax = axes[0, 1]
ax.scatter(df['tsne_1'], df['tsne_2'], c=colors, s=80, edgecolors='black', linewidth=0.5)
for _, r in df.iterrows():
    ax.annotate(str(int(r['Identifier'])), (r['tsne_1'], r['tsne_2']), fontsize=7, ha='center', va='bottom')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE - Chemical Space')

# 3. Similarity heatmap
ax = axes[0, 2]
order = np.argsort(clusters)
sim_ordered = sim_matrix_morgan[np.ix_(order, order)]
im = ax.imshow(sim_ordered, cmap='YlOrRd', vmin=0, vmax=1)
ax.set_title('Morgan Similarity Matrix')
ax.set_xlabel('Compound')
ax.set_ylabel('Compound')
plt.colorbar(im, ax=ax)

# 4. Dendrogram
ax = axes[1, 0]
dendrogram(Z, labels=df['Identifier'].values, leaf_rotation=90, ax=ax)
ax.set_title('Hierarchical Clustering Dendrogram')
ax.set_ylabel('Distance')

# 5. Activity landscape
ax = axes[1, 1]
non_zero = df[df[act_col] > 0]
scatter = ax.scatter(non_zero['pca_1'], non_zero['pca_2'], 
                     c=non_zero['pPotency'], cmap='RdYlBu_r', s=120, 
                     edgecolors='black', linewidth=0.5, vmin=0)
for _, r in non_zero.iterrows():
    ax.annotate(str(int(r['Identifier'])), (r['pca_1'], r['pca_2']), fontsize=8, ha='center', va='bottom')
plt.colorbar(scatter, ax=ax, label='pPotency (-log10[uM])')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Activity Landscape (active compounds)')

# 6. Potency vs Potentiation scatter
ax = axes[1, 2]
ax.scatter(non_zero[act_col], non_zero[pot_col], c='steelblue', s=100, edgecolors='black', linewidth=0.5)
for _, r in non_zero.iterrows():
    ax.annotate(str(int(r['Identifier'])), (r[act_col], r[pot_col]), fontsize=8, ha='center', va='bottom')
ax.set_xlabel('Potency (uM)')
ax.set_ylabel('% Potentiation')
ax.set_title('Potency vs Potentiation')
ax.set_xscale('log')

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, '02_cheminformatics.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Saved figure: {fig_path}")

# ── SAR CLUSTER ANALYSIS ────────────────────────────────────────────────────
print("\n--- SAR Cluster Analysis ---")
for c in range(1, n_clusters + 1):
    members = df[df['cluster'] == c]
    active_in = members[members['is_active'] == 1]
    print(f"\nCluster {c}:")
    print(f"  Size: {len(members)}")
    print(f"  Active: {len(active_in)}")
    print(f"  MW range: {members['MW'].min():.1f} - {members['MW'].max():.1f}")
    print(f"  LogP range: {members['LogP'].min():.2f} - {members['LogP'].max():.2f}")
    for _, r in members.iterrows():
        act_str = f"{r[act_col]} uM, {r[pot_col]}%" if r['is_active'] == 1 else "inactive"
        print(f"    ID {int(r['Identifier'])}: {r['canonical_smiles'][:60]}... [{act_str}]")

# Save exploration results
df.to_csv(os.path.join(OUT_DIR, 'exploratory/cheminformatics_results.csv'), index=False)

print("\n=== STEPS 2-4 COMPLETE ===")
