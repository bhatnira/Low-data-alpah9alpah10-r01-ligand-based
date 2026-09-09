#!/usr/bin/env python3
"""Step 4: Common-Core / Congeneric Analysis, Murcko Scaffolds, MMP Analysis."""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFMCS
from rdkit.Chem import (
    AllChem, rdMolDescriptors, rdFingerprintGenerator,
    Draw, rdmolops, Fragments
)
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# ── CONFIG ──────────────────────────────────────────────────────────────────
PROJECT = '/Users/nb/Documents/ligand-based-modeling/project'
QC_CSV = os.path.join(PROJECT, 'data/qc/data_qc.csv')
EXPLORE_CSV = os.path.join(PROJECT, 'analysis/exploratory/cheminformatics_results.csv')
FIG_DIR = os.path.join(PROJECT, 'figures')
OUT_SAR = os.path.join(PROJECT, 'analysis/SAR')
OUT_MMP = os.path.join(PROJECT, 'analysis/MMP')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_SAR, exist_ok=True)
os.makedirs(OUT_MMP, exist_ok=True)

df = pd.read_csv(EXPLORE_CSV)
df['mol'] = df['canonical_smiles'].apply(Chem.MolFromSmiles)
print(f"Loaded {len(df)} compounds")

act_col = 'Activity (uM)'
pot_col = '%Potentiation'

# ══════════════════════════════════════════════════════════════════════════════
# MCS ANALYSIS - Find Common Scaffold
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MCS (Maximum Common Substructure) ANALYSIS")
print("=" * 70)

# Compute MCS across all 30 molecules
mols = [m for m in df['mol'] if m is not None]
mcs_result = rdFMCS.FindMCS(mols, ringMatchesRingOnly=True, completeRingsOnly=True)
mcs_smarts = mcs_result.smartsString
print(f"\nMCS SMARTS: {mcs_smarts}")
print(f"MCS atoms: {mcs_result.numAtoms}")
print(f"MCS bonds: {mcs_result.numBonds}")

# Parse MCS to get the common scaffold
mcs_mol = Chem.MolFromSmarts(mcs_smarts)
if mcs_mol:
    try:
        print(f"MCS molecular formula: {rdMolDescriptors.CalcMolFormula(mcs_mol)}")
        print(f"MCS MW: {rdMolDescriptors.CalcExactMolWt(mcs_mol):.2f}")
    except:
        print("Could not calculate MCS formula (SMARTS pattern)")

# ══════════════════════════════════════════════════════════════════════════════
# MURCKO SCAFFOLD ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MURCKO SCAFFOLD ANALYSIS")
print("=" * 70)

scaffolds = []
for i, row in df.iterrows():
    mol = row['mol']
    if mol is None:
        scaffolds.append(None)
        continue
    smi = row['canonical_smiles']
    scaffold = MurckoScaffoldSmiles(smi, mol, includeChirality=True)
    scaffolds.append(scaffold)

df['murcko_scaffold'] = scaffolds

print("\nMurcko scaffolds found:")
scaffold_counts = df['murcko_scaffold'].value_counts()
for sc, count in scaffold_counts.items():
    members = df[df['murcko_scaffold'] == sc]
    active_in = members[members['is_active'] == 1]
    print(f"\n  Scaffold: {sc}")
    print(f"  Count: {count}")
    print(f"  Active: {len(active_in)}")
    if len(active_in) > 0:
        print(f"  Active IDs: {active_in['Identifier'].tolist()}")

# Unique scaffolds
n_scaffolds = df['murcko_scaffold'].nunique()
print(f"\nTotal unique Murcko scaffolds: {n_scaffolds}")

# ══════════════════════════════════════════════════════════════════════════════
# SCAFFOLD EXTRACTION & VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SCAFFOLD ANALYSIS")
print("=" * 70)

# Analyze each compound's scaffold relationship
for i, row in df.iterrows():
    mol = row['mol']
    if mol is None:
        continue
    
    # Extract core scaffold by removing side chains
    core = MurckoScaffoldSmiles(row['canonical_smiles'], mol, includeChirality=True)
    df.at[i, 'scaffold_smiles'] = core
    
    # Calculate number of rings in scaffold
    core_mol = Chem.MolFromSmiles(core)
    if core_mol:
        df.at[i, 'n_scaffold_rings'] = core_mol.GetRingInfo().NumRings()
    else:
        df.at[i, 'n_scaffold_rings'] = 0

# Classify by scaffold similarity to active compounds
active_scaffolds = set()
for _, r in df[df['is_active'] == 1].iterrows():
    active_scaffolds.add(r['murcko_scaffold'])

df['has_active_scaffold'] = df['murcko_scaffold'].isin(active_scaffolds).astype(int)
print(f"\nCompounds sharing scaffolds with actives: {df['has_active_scaffold'].sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# MATCHED MOLECULAR PAIR (MMP) ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MATCHED MOLECULAR PAIR (MMP) ANALYSIS")
print("=" * 70)

def find_mmp_pairs(df, sim_threshold=0.7):
    """Find MMP pairs based on high similarity (same scaffold, different substituents)."""
    pairs = []
    n = len(df)
    
    for i in range(n):
        for j in range(i+1, n):
            mol_i = df.iloc[i]['mol']
            mol_j = df.iloc[j]['mol']
            if mol_i is None or mol_j is None:
                continue
            
            # Check if same scaffold
            sc_i = df.iloc[i]['murcko_scaffold']
            sc_j = df.iloc[j]['murcko_scaffold']
            
            if sc_i == sc_j:
                # Calculate Morgan similarity
                fp_i = AllChem.GetMorganFingerprintAsBitVect(mol_i, radius=2, nBits=2048)
                fp_j = AllChem.GetMorganFingerprintAsBitVect(mol_j, radius=2, nBits=2048)
                sim = DataStructs.TanimotoSimilarity(fp_i, fp_j)
                
                if sim >= 0.5:  # Relatively similar but different
                    pairs.append({
                        'idx_i': df.iloc[i]['Identifier'],
                        'idx_j': df.iloc[j]['Identifier'],
                        'smiles_i': df.iloc[i]['canonical_smiles'],
                        'smiles_j': df.iloc[j]['canonical_smiles'],
                        'activity_i': df.iloc[i][act_col],
                        'activity_j': df.iloc[j][act_col],
                        'potentiation_i': df.iloc[i][pot_col],
                        'potentiation_j': df.iloc[j][pot_col],
                        'similarity': sim,
                        'scaffold': sc_i,
                        'is_active_i': df.iloc[i]['is_active'],
                        'is_active_j': df.iloc[j]['is_active']
                    })
    
    return pd.DataFrame(pairs)

mmp_df = find_mmp_pairs(df)
print(f"\nMMP pairs found: {len(mmp_df)}")

if len(mmp_df) > 0:
    # Calculate activity differences
    mmp_df['delta_potency_uM'] = mmp_df['activity_j'] - mmp_df['activity_i']
    mmp_df['delta_potentiation'] = mmp_df['potentiation_j'] - mmp_df['potentiation_i']
    
    # Activity change significance
    mmp_df['activity_change'] = 'no_change'
    for idx, row in mmp_df.iterrows():
        if row['activity_i'] > 0 and row['activity_j'] > 0:
            # Both active - check fold change
            fold = row['activity_j'] / row['activity_i'] if row['activity_i'] > 0 else float('inf')
            if fold > 2:
                mmp_df.at[idx, 'activity_change'] = 'decreased_potency'
            elif fold < 0.5:
                mmp_df.at[idx, 'activity_change'] = 'increased_potency'
        elif row['activity_i'] > 0 and row['activity_j'] == 0:
            mmp_df.at[idx, 'activity_change'] = 'lost_activity'
        elif row['activity_i'] == 0 and row['activity_j'] > 0:
            mmp_df.at[idx, 'activity_change'] = 'gained_activity'
    
    print("\nMMP Activity Changes:")
    print(mmp_df[['idx_i', 'idx_j', 'activity_i', 'activity_j', 
                   'delta_potency_uM', 'activity_change', 'similarity']].to_string())
    
    mmp_df.to_csv(os.path.join(OUT_MMP, 'mmp_pairs.csv'), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# R-GROUP DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("R-GROUP / SUBSTITUENT ANALYSIS")
print("=" * 70)

# Identify common patterns and substituents
# Analyze the key structural features of active vs inactive

# For active compounds, identify key features
active_compounds = df[df['is_active'] == 1].copy()
inactive_compounds = df[df['is_active'] == 0].copy()

print("\nActive compounds key features:")
for _, r in active_compounds.iterrows():
    mol = r['mol']
    if mol is None:
        continue
    
    # Check for common substructures
    has_nitrogen = mol.GetSubstructMatch(Chem.MolFromSmarts('[#7]')) != ()
    has_bromine = mol.GetSubstructMatch(Chem.MolFromSmarts('[#35]')) != ()
    has_alkyne = mol.GetSubstructMatch(Chem.MolFromSmarts('C#C')) != ()
    has_benzyl = mol.GetSubstructMatch(Chem.MolFromSmarts('c1ccccc1C')) != ()
    has_acetal = mol.GetSubstructMatch(Chem.MolFromSmarts('[OR0][CR0][OR0]')) != ()
    has_lactone = mol.GetSubstructMatch(Chem.MolFromSmarts('C(=O)O[C,C]')) != ()
    
    print(f"\n  ID {int(r['Identifier'])}: {r['canonical_smiles'][:70]}...")
    print(f"    Potency: {r[act_col]} uM, Potentiation: {r[pot_col]}%")
    print(f"    MW: {r['MW']:.1f}, LogP: {r['LogP']:.2f}")
    print(f"    N: {has_nitrogen}, Br: {has_bromine}, Alkyne: {has_alkyne}")
    print(f"    Benzyl: {has_benzyl}, Acetal: {has_acetal}, Lactone: {has_lactone}")
    print(f"    HBD: {r['HBD']}, HBA: {r['HBA']}, TPSA: {r['TPSA']:.1f}")

# Feature frequency comparison
print("\n--- Feature Frequency: Active vs Inactive ---")
features = {
    'Nitrogen': '[#7]',
    'Bromine': '[#35]',
    'Alkyne': 'C#C',
    'Benzyl': 'c1ccccc1C',
    'Acetal': '[OR0][CR0][OR0]',
    'Lactone': 'C(=O)O[C,C]',
    'Phenol': 'c1ccc(O)cc1',
    'Ether': '[C!H0]O[C!H0]',
    'Alcohol': '[CX4][OH]',
    'Chlorine': '[#17]'
}

feature_table = []
for feat_name, smarts in features.items():
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        continue
    n_active = sum(1 for _, r in active_compounds.iterrows() if r['mol'] and r['mol'].GetSubstructMatch(patt))
    n_inactive = sum(1 for _, r in inactive_compounds.iterrows() if r['mol'] and r['mol'].GetSubstructMatch(patt))
    feature_table.append({
        'Feature': feat_name,
        'SMARTS': smarts,
        'Active': n_active,
        'Active_%': f"{n_active}/{len(active_compounds)} ({100*n_active/len(active_compounds):.0f}%)" if len(active_compounds) > 0 else "0%",
        'Inactive': n_inactive,
        'Inactive_%': f"{n_inactive}/{len(inactive_compounds)} ({100*n_inactive/len(inactive_compounds):.0f}%)" if len(inactive_compounds) > 0 else "0%"
    })

feat_df = pd.DataFrame(feature_table)
print(feat_df.to_string(index=False))
feat_df.to_csv(os.path.join(OUT_SAR, 'feature_frequency.csv'), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL HYPOTHESIS GENERATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STRUCTURAL SAR HYPOTHESES")
print("=" * 70)

# Most potent compound analysis (ID 12: 0.1981 uM)
most_potent = df[df['Identifier'] == 12].iloc[0]
print(f"\nMost potent compound: ID 12 ({most_potent[act_col]} uM)")
print(f"  SMILES: {most_potent['canonical_smiles']}")
print(f"  Key features:")
print(f"    - Ethynyl group (C#C): likely important for activity")
print(f"    - Stereochemistry: [C@@H] and [C@H] - defined configuration")
print(f"    - Tether length: 2-carbon linker to terminal alkyne")
print(f"    - Lactone core: consistent across series")

# Second most potent (ID 25: 2.63 uM)
second_potent = df[df['Identifier'] == 25].iloc[0]
print(f"\nSecond potent compound: ID 25 ({second_potent[act_col]} uM)")
print(f"  SMILES: {second_potent['canonical_smiles']}")
print(f"  Key features:")
print(f"    - Bromine atom: electron-withdrawing, hydrophobic")
print(f"    - Stereochemistry: defined")
print(f"    - Similar lactone core")

# Compare active vs inactive patterns
print("\n\n--- SAR PATTERNS ---")
print("1. Active compounds have relatively low LogP range (-1.4 to 2.7)")
print("2. Active compounds cluster in MW range 176-300 Da")
print("3. The most potent compound (ID 12) has an ethynyl group")
print("4. The second most potent (ID 25) has a bromine atom")
print("5. Many inactive compounds share similar scaffolds to actives")
print("   suggesting substituent effects are critical")
print("6. Stereochemistry is consistently defined in active compounds")
print("7. Hydrogen bonding capability (HBD/HBA) varies among actives")

# Save comprehensive SAR report
report = pd.DataFrame([{
    'Finding': 'MCS Core',
    'Detail': mcs_smarts,
    'Confidence': 'High'
}, {
    'Finding': 'Active scaffold diversity',
    'Detail': f'{n_scaffolds} unique Murcko scaffolds in 30 compounds',
    'Confidence': 'High'
}, {
    'Finding': 'Most potent compound',
    'Detail': 'ID 12 (0.1981 uM) - ethynyl group + defined stereo',
    'Confidence': 'Experimental'
}, {
    'Finding': 'Key pharmacophore hints',
    'Detail': 'H-bond acceptor (lactone), hydrophobic region, stereochemistry',
    'Confidence': 'Suggestive'
}])
report.to_csv(os.path.join(OUT_SAR, 'sar_hypotheses.csv'), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Scaffold distribution
ax = axes[0, 0]
sc_data = df.groupby('murcko_scaffold').agg({
    'Identifier': 'count',
    'is_active': 'sum'
}).rename(columns={'Identifier': 'count', 'is_active': 'n_active'})
sc_data = sc_data.sort_values('count', ascending=False).head(8)
bars = ax.bar(range(len(sc_data)), sc_data['count'], color='steelblue', edgecolor='black')
ax.bar(range(len(sc_data)), sc_data['n_active'], color='red', edgecolor='black', alpha=0.7)
ax.set_xticks(range(len(sc_data)))
ax.set_xticklabels([s[:30] for s in sc_data.index], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Count')
ax.set_title('Murcko Scaffold Distribution')
ax.legend(['Total', 'Active'])

# 2. MW vs LogP by scaffold
ax = axes[0, 1]
for sc in df['murcko_scaffold'].unique():
    mask = df['murcko_scaffold'] == sc
    act_mask = mask & (df['is_active'] == 1)
    inact_mask = mask & (df['is_active'] == 0)
    ax.scatter(df.loc[inact_mask, 'MW'], df.loc[inact_mask, 'LogP'], 
               s=60, alpha=0.5, edgecolors='black', linewidth=0.5)
    if act_mask.sum() > 0:
        ax.scatter(df.loc[act_mask, 'MW'], df.loc[act_mask, 'LogP'],
                   s=120, marker='*', edgecolors='black', linewidth=0.5, zorder=5)
ax.set_xlabel('MW')
ax.set_ylabel('LogP')
ax.set_title('Chemical Space by Scaffold')
ax.legend(['Inactive', 'Active (star)'])

# 3. Feature heatmap for active compounds
ax = axes[1, 0]
active_features = []
for _, r in active_compounds.iterrows():
    mol = r['mol']
    if mol is None:
        continue
    feats = {}
    for feat_name, smarts in features.items():
        patt = Chem.MolFromSmarts(smarts)
        feats[feat_name] = 1 if patt and mol.GetSubstructMatch(patt) else 0
    feats['ID'] = str(int(r['Identifier']))
    feats['pPotency'] = r.get('pPotency', 0)
    active_features.append(feats)

if active_features:
    feat_matrix = pd.DataFrame(active_features).set_index('ID')
    sns.heatmap(feat_matrix.drop(columns=['pPotency'], errors='ignore').T, 
                cmap='YlOrRd', annot=True, fmt='g', ax=ax)
    ax.set_title('Feature Presence in Active Compounds')

# 4. Potency vs key descriptor
ax = axes[1, 1]
non_zero = df[df[act_col] > 0]
ax.scatter(non_zero['MW'], non_zero['pPotency'], c='steelblue', s=100, 
           edgecolors='black', linewidth=0.5)
for _, r in non_zero.iterrows():
    ax.annotate(str(int(r['Identifier'])), (r['MW'], r['pPotency']), 
                fontsize=8, ha='center', va='bottom')
ax.set_xlabel('MW')
ax.set_ylabel('pPotency (-log10[uM])')
ax.set_title('MW vs Potency')

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, '03_scaffold_sar.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nSaved figure: {fig_path}")

# Save scaffold data
df_scaffold = df[['Identifier', 'canonical_smiles', act_col, pot_col, 'is_active',
                   'murcko_scaffold', 'has_active_scaffold', 'n_scaffold_rings']].copy()
df_scaffold.to_csv(os.path.join(OUT_SAR, 'scaffold_analysis.csv'), index=False)

print("\n=== STEP 4 COMPLETE ===")
