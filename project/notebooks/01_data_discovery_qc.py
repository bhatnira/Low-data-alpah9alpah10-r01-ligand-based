#!/usr/bin/env python3
"""Step 1: Data Discovery & Quality Control for α9α10 nAChR PAM Dataset."""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    Descriptors, AllChem, Draw, PandasTools, rdMolDescriptors,
    Fragments, rdFingerprintGenerator
)
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────────────────────────
PROJECT = '/Users/nb/Documents/ligand-based-modeling/project'
RAW_CSV = os.path.join(PROJECT, 'data/raw/modulator-dataset-a9a10.csv')
OUT_QC = os.path.join(PROJECT, 'data/qc/data_qc.csv')
OUT_PROCESSED = os.path.join(PROJECT, 'data/processed/data_cleaned.csv')
FIG_DIR = os.path.join(PROJECT, 'figures')
os.makedirs(os.path.dirname(OUT_QC), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── LOAD DATA ───────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_CSV)
print(f"Loaded {len(df)} rows from {RAW_CSV}")
print(f"Columns: {list(df.columns)}")
print(df.head(10).to_string())
print("\n--- Data types ---")
print(df.dtypes)
print("\n--- Missing values ---")
print(df.isnull().sum())
print("\n--- Basic stats ---")
print(df.describe())

# ── DATA DICTIONARY ─────────────────────────────────────────────────────────
data_dict = pd.DataFrame({
    'Field': ['Identifier', 'Smiles', 'Activity (uM)', '%Potentiation'],
    'Meaning': [
        'Unique compound ID',
        'SMILES string representation',
        'Potency in micromolar (lower = more potent)',
        'Percent potentiation (efficacy measure)'
    ],
    'Data type': [str, str, 'float64', 'float64'],
    'Missing values': [
        df['Identifier'].isna().sum(),
        df['Smiles'].isna().sum(),
        df['Activity (uM)'].isna().sum(),
        df['%Potentiation'].isna().sum()
    ]
})
print("\n=== DATA DICTIONARY ===")
print(data_dict.to_string(index=False))

# ── SMILES PARSING & VALIDATION ─────────────────────────────────────────────
print("\n=== SMILES PARSING ===")
parsed_mols = []
valid_smiles = []
invalid_idx = []

for i, smi in enumerate(df['Smiles']):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        invalid_idx.append(i)
        parsed_mols.append(None)
        valid_smiles.append(smi)
    else:
        parsed_mols.append(mol)
        valid_smiles.append(Chem.MolToSmiles(mol))

df['parsed_mol'] = parsed_mols
df['canonical_smiles'] = valid_smiles

print(f"Total compounds: {len(df)}")
print(f"Valid SMILES: {len(df) - len(invalid_idx)}")
print(f"Invalid SMILES: {len(invalid_idx)}")
if invalid_idx:
    print(f"Invalid indices: {invalid_idx}")
    print(df.loc[invalid_idx, ['Identifier', 'Smiles']])

# ── DUPLICATE DETECTION ─────────────────────────────────────────────────────
print("\n=== DUPLICATE DETECTION ===")
# Exact structure duplicates (after canonicalization)
dup_mask = df.duplicated(subset=['canonical_smiles'], keep=False)
n_dup = dup_mask.sum()
print(f"Exact canonical SMILES duplicates: {n_dup} compounds in {df['canonical_smiles'].nunique()} unique structures")
if n_dup > 0:
    for smi in df.loc[dup_mask, 'canonical_smiles'].unique():
        dup_rows = df[df['canonical_smiles'] == smi]
        print(f"  Dup SMILES: {smi}")
        print(f"    IDs: {dup_rows['Identifier'].tolist()}")

# Check for stereochemical duplicates (same connectivity, different stereo)
connectivities = []
for i, mol in enumerate(df['parsed_mol']):
    if mol is not None:
        c = Chem.MolToSmiles(mol, isomericSmiles=False)
        connectivities.append(c)
    else:
        connectivities.append(None)
df['connectivity'] = connectivities

conn_mask = df.duplicated(subset=['connectivity'], keep=False)
n_conn_dup = conn_mask.sum()
print(f"\nConnectivity duplicates (ignoring stereo): {n_conn_dup} compounds")
if n_conn_dup > 0:
    for conn in df.loc[conn_mask, 'connectivity'].unique():
        if conn is None:
            continue
        rows = df[df['connectivity'] == conn]
        print(f"  Connectivity: {conn}")
        print(f"    IDs: {rows['Identifier'].tolist()}")
        for _, r in rows.iterrows():
            print(f"      Stereo SMILES: {r['canonical_smiles']}")

# ── MOLECULAR DESCRIPTORS ───────────────────────────────────────────────────
print("\n=== CALCULATING MOLECULAR DESCRIPTORS ===")

descriptor_fns = {
    'MW': Descriptors.MolWt,
    'LogP': Descriptors.MolLogP,
    'TPSA': Descriptors.TPSA,
    'HBD': Descriptors.NumHDonors,
    'HBA': Descriptors.NumHAcceptors,
    'RotBonds': Descriptors.NumRotatableBonds,
    'RingCount': Descriptors.RingCount,
    'AromaticRings': Descriptors.NumAromaticRings,
    'FractionCSP3': Descriptors.FractionCSP3,
    'FormalCharge': Chem.GetFormalCharge,
    'HeavyAtomCount': Descriptors.HeavyAtomCount,
    'NumHeteroatoms': Descriptors.NumHeteroatoms,
    'NumValenceElectrons': Descriptors.NumValenceElectrons,
    'NumAmideBonds': Descriptors.NumAmideBonds,
    'MolFormula': rdMolDescriptors.CalcMolFormula,
    'NumStereocenters': rdMolDescriptors.CalcNumAtomStereoCenters,
    'NumUnspecifiedStereocenters': rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters,
    'NumBonds': lambda m: m.GetNumBonds() if m else 0,
    'NumAliphaticRings': Descriptors.NumAliphaticRings,
    'NumSaturatedRings': Descriptors.NumSaturatedRings,
    'LabuteASA': Descriptors.LabuteASA,
    'BertzCT': Descriptors.BertzCT,
}

for name, fn in descriptor_fns.items():
    df[name] = df['parsed_mol'].apply(lambda m: fn(m) if m is not None else np.nan)

# ── STEREOCHEMISTRY ANALYSIS ────────────────────────────────────────────────
print("\n=== STEREOCHEMISTRY ANALYSIS ===")

def analyze_stereo(mol):
    if mol is None:
        return {}
    info = {}
    info['has_stereo'] = rdMolDescriptors.CalcNumAtomStereoCenters(mol) > 0
    info['n_stereo_centers'] = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    info['n_unspecified'] = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
    info['n_specified'] = info['n_stereo_centers'] - info['n_unspecified']
    
    # Check for E/Z bonds
    n_ez = 0
    for bond in mol.GetBonds():
        if bond.GetStereo() in [Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ]:
            n_ez += 1
    info['n_ez_bonds'] = n_ez
    
    # Count undefined stereo
    info['has_undefined_stereo'] = info['n_unspecified'] > 0
    
    # Flag
    if info['n_stereo_centers'] > 0:
        info['stereo_status'] = 'defined' if info['n_unspecified'] == 0 else 'undefined/mixed'
    elif n_ez > 0:
        info['stereo_status'] = 'E/Z defined'
    else:
        info['stereo_status'] = 'no_stereo'
    
    return info

stereo_data = df['parsed_mol'].apply(analyze_stereo)
stereo_df = pd.DataFrame(stereo_data.tolist())
for col in stereo_df.columns:
    df[f'stereo_{col}'] = stereo_df[col].values

print(df[['Identifier', 'canonical_smiles', 'stereo_stereo_status', 
          'stereo_n_stereo_centers', 'stereo_n_specified', 'stereo_n_unspecified',
          'stereo_n_ez_bonds']].to_string())

# ── DUAL ACTIVITY ANALYSIS ──────────────────────────────────────────────────
print("\n=== ACTIVITY ANALYSIS ===")
act_col = 'Activity (uM)'
pot_col = '%Potentiation'

print(f"\n{act_col}:")
print(f"  Range: {df[act_col].min()} - {df[act_col].max()} uM")
print(f"  Mean: {df[act_col].mean():.1f} uM")
print(f"  Median: {df[act_col].median():.1f} uM")
print(f"  Zero values: {(df[act_col] == 0).sum()}")
print(f"  Non-zero values: {(df[act_col] > 0).sum()}")

print(f"\n{pot_col}:")
print(f"  Range: {df[pot_col].min()} - {df[pot_col].max()} %")
print(f"  Mean: {df[pot_col].mean():.1f} %")
print(f"  Non-zero potentiation: {(df[pot_col] > 0).sum()}")

# Classify: active if potency > threshold and potentiation > threshold
# A compound with Activity(uM)=0 likely means "inactive" or "not measured"
# A compound with >0 potency AND >0% potentiation is likely active
df['is_active'] = ((df[act_col] > 0) & (df[pot_col] > 0)).astype(int)
print(f"\nActivity classification:")
print(f"  Active (>0 uM potency, >0% pot): {df['is_active'].sum()}")
print(f"  Inactive/unknown: {(df['is_active'] == 0).sum()}")

# Rank by potency (lower UM = more potent)
df['potency_rank'] = df[act_col].replace(0, np.nan).rank(ascending=True, method='min')
print(f"\nPotency ranking (non-zero compounds):")
active_df = df[df[act_col] > 0].sort_values(act_col)
for _, r in active_df.iterrows():
    print(f"  Rank {int(r['potency_rank'])}: ID={r['Identifier']}, {r[act_col]} uM, {r[pot_col]}% pot")

# ── GENERATE QC OUTPUT ──────────────────────────────────────────────────────
qc_cols = [
    'Identifier', 'Smiles', 'canonical_smiles', act_col, pot_col,
    'is_active', 'potency_rank', 'MW', 'LogP', 'TPSA', 'HBD', 'HBA',
    'RotBonds', 'RingCount', 'AromaticRings', 'FractionCSP3', 'FormalCharge',
    'HeavyAtomCount', 'NumHeteroatoms', 'MolFormula',
    'stereo_has_stereo', 'stereo_n_stereo_centers', 'stereo_n_specified',
    'stereo_n_unspecified', 'stereo_n_ez_bonds', 'stereo_stereo_status',
    'stereo_has_undefined_stereo', 'connectivity'
]
df_qc = df[qc_cols].copy()
df_qc.to_csv(OUT_QC, index=False)
print(f"\nSaved QC data to {OUT_QC}")

# ── SUMMARY REPORT ──────────────────────────────────────────────────────────
report_lines = []
report_lines.append("=" * 70)
report_lines.append("DATA QUALITY CONTROL REPORT")
report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("=" * 70)
report_lines.append(f"\nDataset: {RAW_CSV}")
report_lines.append(f"Total rows: {len(df)}")
report_lines.append(f"Unique compounds (by ID): {df['Identifier'].nunique()}")
report_lines.append(f"Unique canonical SMILES: {df['canonical_smiles'].nunique()}")
report_lines.append(f"Valid SMILES parsed: {len(df) - len(invalid_idx)}")
report_lines.append(f"Invalid SMILES: {len(invalid_idx)}")
report_lines.append(f"\nActivity Summary:")
report_lines.append(f"  Potency (uM) range: {df[act_col].min()} - {df[act_col].max()}")
report_lines.append(f"  Active compounds: {df['is_active'].sum()}")
report_lines.append(f"  Inactive/unknown: {(df['is_active'] == 0).sum()}")
report_lines.append(f"\nPotency-active compounds (>0 uM, >0% pot):")
for _, r in active_df.iterrows():
    report_lines.append(f"  ID {r['Identifier']}: {r[act_col]} uM, {r[pot_col]}% pot, SMILES={r['canonical_smiles']}")
report_lines.append(f"\nStereochemistry:")
n_stereo = df['stereo_has_stereo'].sum()
report_lines.append(f"  Compounds with stereocenters: {n_stereo}")
n_undef = df['stereo_has_undefined_stereo'].sum()
report_lines.append(f"  Compounds with undefined stereo: {n_undef}")
n_ez = (df['stereo_n_ez_bonds'] > 0).sum()
report_lines.append(f"  Compounds with E/Z bonds: {n_ez}")
report_lines.append(f"\nDuplicates:")
report_lines.append(f"  Exact canonical SMILES duplicates: {n_dup}")
report_lines.append(f"  Connectivity duplicates (ignoring stereo): {n_conn_dup}")
report_lines.append(f"\nMissing Values:")
for col in ['Smiles', act_col, pot_col]:
    report_lines.append(f"  {col}: {df[col].isna().sum()}")
report_lines.append(f"\nMolecular Properties Summary (all compounds):")
for col in ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'FractionCSP3']:
    report_lines.append(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, range=[{df[col].min():.2f}, {df[col].max():.2f}]")

report_text = '\n'.join(report_lines)
report_path = os.path.join(PROJECT, 'reports/qc_report.txt')
with open(report_path, 'w') as f:
    f.write(report_text)
print(f"\nSaved QC report to {report_path}")
print("\n" + report_text)

# ── FIGURES ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Activity distribution
ax = axes[0, 0]
active_mask = df[act_col] > 0
ax.hist(df.loc[active_mask, act_col].values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Potency (uM)')
ax.set_ylabel('Count')
ax.set_title('Potency Distribution (active compounds)')

# 2. % Potentiation distribution
ax = axes[0, 1]
pot_mask = df[pot_col] > 0
ax.hist(df.loc[pot_mask, pot_col].values, bins=20, color='coral', edgecolor='black', alpha=0.7)
ax.set_xlabel('% Potentiation')
ax.set_ylabel('Count')
ax.set_title('% Potentiation Distribution')

# 3. MW vs LogP colored by activity
ax = axes[0, 2]
colors = df['is_active'].map({1: 'red', 0: 'gray'})
ax.scatter(df['MW'], df['LogP'], c=colors, s=60, edgecolors='black', linewidth=0.5, alpha=0.7)
ax.set_xlabel('Molecular Weight')
ax.set_ylabel('LogP')
ax.set_title('MW vs LogP')
ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Active'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Inactive')],
          loc='best')

# 4. TPSA vs HBD
ax = axes[1, 0]
ax.scatter(df['TPSA'], df['HBD'], c=colors, s=60, edgecolors='black', linewidth=0.5, alpha=0.7)
ax.set_xlabel('TPSA')
ax.set_ylabel('HBD')
ax.set_title('TPSA vs HBD')

# 5. Potency vs %Potentiation
ax = axes[1, 1]
non_zero = df[(df[act_col] > 0) & (df[pot_col] > 0)]
ax.scatter(non_zero[act_col], non_zero[pot_col], c='steelblue', s=80, edgecolors='black', linewidth=0.5)
for _, r in non_zero.iterrows():
    ax.annotate(str(int(r['Identifier'])), (r[act_col], r[pot_col]), fontsize=8, ha='center', va='bottom')
ax.set_xlabel('Potency (uM)')
ax.set_ylabel('% Potentiation')
ax.set_title('Potency vs % Potentiation')

# 6. Fraction CSP3
ax = axes[1, 2]
ax.hist(df['FractionCSP3'].dropna(), bins=15, color='green', edgecolor='black', alpha=0.7)
ax.set_xlabel('Fraction CSP3')
ax.set_ylabel('Count')
ax.set_title('Fraction CSP3 Distribution')

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, '01_data_overview.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nSaved overview figure to {fig_path}")

# ── SAVE PROCESSED DATA ─────────────────────────────────────────────────────
drop_cols = ['parsed_mol']
df.drop(columns=drop_cols, inplace=True, errors='ignore')
df.to_csv(OUT_PROCESSED, index=False)
print(f"Saved processed data to {OUT_PROCESSED}")
print("\n=== STEP 1 COMPLETE ===")
