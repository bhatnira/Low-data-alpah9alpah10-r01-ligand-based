#!/usr/bin/env python3
"""Steps 5-7: Few-Shot Learning, Explainable SAR, TAF Discovery, Stereochemistry Analysis."""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

PROJECT = '/Users/nb/Documents/ligand-based-modeling/project'
EXPLORE_CSV = os.path.join(PROJECT, 'analysis/exploratory/cheminformatics_results.csv')
FIG_DIR = os.path.join(PROJECT, 'figures')
OUT_TAF = os.path.join(PROJECT, 'analysis/TAF')
OUT_STEREO = os.path.join(PROJECT, 'analysis/stereochemistry')
OUT_3D = os.path.join(PROJECT, 'analysis/3D')
for d in [FIG_DIR, OUT_TAF, OUT_STEREO, OUT_3D]:
    os.makedirs(d, exist_ok=True)

df = pd.read_csv(EXPLORE_CSV)
df['mol'] = df['canonical_smiles'].apply(Chem.MolFromSmiles)
act_col = 'Activity (uM)'
pot_col = '%Potentiation'
print(f"Loaded {len(df)} compounds")

# Build feature matrix
descriptor_cols = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 
                   'FractionCSP3', 'RingCount', 'AromaticRings', 'HeavyAtomCount']

fps_list = []
for mol in df['mol']:
    if mol is None:
        fps_list.append(np.zeros(2048, dtype=np.float64))
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        arr = np.zeros(2048, dtype=np.float64)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps_list.append(arr)
fps_matrix = np.vstack(fps_list)
desc_matrix = df[descriptor_cols].fillna(0).values.astype(np.float64)
X = np.hstack([fps_matrix, desc_matrix])
y_binary = df['is_active'].values
print(f"Feature matrix: {X.shape}, Active: {y_binary.sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: FEW-SHOT LEARNING & LOO CV
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: FEW-SHOT LEARNING & VALIDATION")
print("=" * 70)

models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
    'RidgeClassifier': RidgeClassifier(alpha=1.0, class_weight='balanced')
}

loo = LeaveOneOut()
cv_results = {}

for name, model in models.items():
    preds = []
    true = []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_binary[train_idx], y_binary[test_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model.fit(X_train_s, y_train)
        preds.append(model.predict(X_test_s)[0])
        true.append(y_test[0])
    acc = accuracy_score(true, preds)
    cv_results[name] = acc
    print(f"  {name}: LOO accuracy = {acc:.3f} ({sum(p==t for p,t in zip(preds,true))}/{len(true)})")

# Leave-one-active-out
print("\n--- Leave-One-Active-Out ---")
active_indices = np.where(y_binary == 1)[0]
for active_idx in active_indices:
    cid = int(df.iloc[active_idx]['Identifier'])
    mask = np.ones(len(df), dtype=bool)
    mask[active_idx] = False
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X[mask])
    X_te_s = scaler.transform(X[~mask])
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_tr_s, y_binary[mask])
    pred = rf.predict(X_te_s)[0]
    print(f"  Leave out ID {cid}: true=1, pred={pred}, correct={pred==1}")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE (using full model, reduced permutation)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Feature Importance ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf.fit(X_scaled, y_binary)

# Use RF built-in importance for speed
rf_importances = rf.feature_importances_
desc_importances = rf_importances[2048:]
fp_importances = rf_importances[:2048]

top_desc_idx = np.argsort(desc_importances)[::-1][:len(descriptor_cols)]
print("\nDescriptor importance (RF Gini):")
for idx in top_desc_idx:
    print(f"  {descriptor_cols[idx]}: {desc_importances[idx]:.4f}")

top_fp_idx = np.argsort(fp_importances)[::-1][:10]
print("\nTop FP bits:")
for idx in top_fp_idx:
    if fp_importances[idx] > 0.001:
        print(f"  Bit {idx}: {fp_importances[idx]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: COUNTERFACTUAL & ABLATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: COUNTERFACTUAL & ABLATION ANALYSIS")
print("=" * 70)

def mol_to_fp_array(mol, nbits=2048):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nbits)
    fp_arr = np.zeros(nbits, dtype=np.float64)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    desc = np.array([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol), Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol), Descriptors.RingCount(mol),
        Descriptors.NumAromaticRings(mol), Descriptors.HeavyAtomCount(mol)
    ], dtype=np.float64)
    return np.hstack([fp_arr, desc])

counterfactual_results = []
most_potent = df[df['Identifier'] == 12].iloc[0]

# CF1: Remove ethynyl
cf1_smi = most_potent['canonical_smiles'].replace('C#CCOC', 'CCOC')
cf1_mol = Chem.MolFromSmiles(cf1_smi)
if cf1_mol:
    fp_cf = mol_to_fp_array(cf1_mol)
    pred = rf.predict(scaler.transform(fp_cf.reshape(1, -1)))[0]
    prob = rf.predict_proba(scaler.transform(fp_cf.reshape(1, -1)))[0].max()
    counterfactual_results.append({'compound': 'ID 12', 'modification': 'Remove ethynyl→methyl', 'predicted_active': bool(pred), 'confidence': prob})
    print(f"  CF1 (remove ethynyl): pred={pred}, conf={prob:.3f}")

# CF2: Remove stereo
cf2_smi = most_potent['canonical_smiles'].replace('@@', '@@@@').replace('@', '').replace('@@@@', '@@')
cf2_mol = Chem.MolFromSmiles(cf2_smi)
if cf2_mol:
    fp_cf = mol_to_fp_array(cf2_mol)
    pred = rf.predict(scaler.transform(fp_cf.reshape(1, -1)))[0]
    prob = rf.predict_proba(scaler.transform(fp_cf.reshape(1, -1)))[0].max()
    counterfactual_results.append({'compound': 'ID 12', 'modification': 'Remove stereo', 'predicted_active': bool(pred), 'confidence': prob})
    print(f"  CF2 (remove stereo): pred={pred}, conf={prob:.3f}")

# CF3: Remove bromine from ID 25
potent25 = df[df['Identifier'] == 25].iloc[0]
cf3_smi = potent25['canonical_smiles'].replace('Br', 'CC')
cf3_mol = Chem.MolFromSmiles(cf3_smi)
if cf3_mol:
    fp_cf = mol_to_fp_array(cf3_mol)
    pred = rf.predict(scaler.transform(fp_cf.reshape(1, -1)))[0]
    prob = rf.predict_proba(scaler.transform(fp_cf.reshape(1, -1)))[0].max()
    counterfactual_results.append({'compound': 'ID 25', 'modification': 'Remove Br→ethyl', 'predicted_active': bool(pred), 'confidence': prob})
    print(f"  CF3 (remove Br): pred={pred}, conf={prob:.3f}")

# Feature ablation (only descriptors for speed)
print("\n--- Feature Ablation ---")
baseline_acc = rf.score(X_scaled, y_binary)
ablation_results = []
for i, desc_name in enumerate(descriptor_cols):
    X_ablated = X_scaled.copy()
    X_ablated[:, 2048 + i] = 0
    acc_ablated = rf.score(X_ablated, y_binary)
    ablation_results.append({
        'feature': desc_name,
        'ablated_accuracy': acc_ablated,
        'accuracy_drop': baseline_acc - acc_ablated
    })
    print(f"  Ablate {desc_name}: acc={acc_ablated:.3f} (drop={baseline_acc-acc_ablated:.4f})")

ablation_df = pd.DataFrame(ablation_results).sort_values('accuracy_drop', ascending=False)
ablation_df.to_csv(os.path.join(OUT_TAF, 'feature_ablation.csv'), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# TAF DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TRANSFERABLE ACTIVITY FEATURE (TAF) DISCOVERY")
print("=" * 70)

active_mols = [m for m in df[df['is_active'] == 1]['mol'] if m]
inactive_mols = [m for m in df[df['is_active'] == 0]['mol'] if m]

taf_patterns = {
    'TAF-1: Lactone ring': 'C(=O)O[C,c]1',
    'TAF-2: Hydroxyl group': '[CX4][OH]',
    'TAF-3: Terminal alkyne': 'C#C',
    'TAF-4: Bromine substituent': '[Br]',
    'TAF-5: Benzyl/aromatic side chain': 'c1ccccc1C',
    'TAF-6: 1,3-dioxolane ring': 'C1OCCO1',
    'TAF-7: Enol/enolate': 'C=C(O)',
    'TAF-8: Defined stereocenter': '[C@H]',
    'TAF-9: Small alkyl chain': 'CCC',
    'TAF-10: Carboxylic acid/ester': 'C(=O)O',
}

taf_evidence = []
for taf_name, smarts in taf_patterns.items():
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        continue
    n_active = sum(1 for m in active_mols if m.GetSubstructMatch(patt))
    n_inactive = sum(1 for m in inactive_mols if m.GetSubstructMatch(patt))
    table = np.array([[n_active, len(active_mols)-n_active], [n_inactive, len(inactive_mols)-n_inactive]])
    _, pval = stats.fisher_exact(table)
    enrichment = (n_active/len(active_mols)) / (n_inactive/len(inactive_mols) + 1e-10)
    evidence = 'Strong' if pval < 0.05 and enrichment > 1.5 else 'Moderate' if enrichment > 1 else 'Weak'
    taf_evidence.append({
        'TAF': taf_name, 'SMARTS': smarts,
        'Active': f"{n_active}/{len(active_mols)}", 'Inactive': f"{n_inactive}/{len(inactive_mols)}",
        'Enrichment': round(enrichment, 2), 'P_value': round(pval, 4), 'Evidence': evidence
    })
    print(f"  {taf_name}: active={n_active}/{len(active_mols)}, inactive={n_inactive}/{len(inactive_mols)}, enrich={enrichment:.2f}, p={pval:.4f} [{evidence}]")

taf_df = pd.DataFrame(taf_evidence)
taf_df.to_csv(os.path.join(OUT_TAF, 'taf_evidence.csv'), index=False)

# TAF Blueprint
blueprint = pd.DataFrame([
    {'TAF': 'TAF-1: Lactone Core', 'Required': True, 'Evidence': '100% actives', '3D_geometry': 'Planar 5-membered ring', 'Stereo_dependence': 'Indirect', 'Confidence': 'High'},
    {'TAF': 'TAF-2: Hydroxyl Group', 'Required': True, 'Evidence': '71% actives, enriched', '3D_geometry': 'OH vector ~109°', 'Stereo_dependence': 'R/S affects orientation', 'Confidence': 'Moderate'},
    {'TAF': 'TAF-3: Electron-Withdrawing Group', 'Required': True, 'Evidence': 'Top 2 potent (Br/alkyne)', '3D_geometry': '3-5 Å from core', 'Stereo_dependence': 'Position dependent', 'Confidence': 'Strong'},
    {'TAF': 'TAF-4: Defined Stereochemistry', 'Required': True, 'Evidence': '7/7 active defined', '3D_geometry': 'Specific R/S config', 'Stereo_dependence': 'Critical', 'Confidence': 'Very High'},
    {'TAF': 'TAF-5: MW 175-300 Da', 'Required': False, 'Evidence': 'All actives in range', '3D_geometry': 'Size for pocket', 'Stereo_dependence': 'N/A', 'Confidence': 'Moderate'},
])
blueprint.to_csv(os.path.join(OUT_TAF, 'taf_blueprint.csv'), index=False)
print("\nTAF Blueprint:")
print(blueprint.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: STEREOCHEMISTRY & 3D ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: STEREOCHEMISTRY & 3D ANALYSIS")
print("=" * 70)

def gen_3d(mol):
    if mol is None: return {}
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    cid = AllChem.EmbedMolecule(mol_h, params)
    if cid == -1: return {}
    try: AllChem.MMFFOptimizeMolecule(mol_h, confId=cid)
    except: pass
    d = {}
    for name, fn in [('Asphericity', Descriptors3D.Asphericity), ('Eccentricity', Descriptors3D.Eccentricity),
                     ('NPR1', Descriptors3D.NPR1), ('NPR2', Descriptors3D.NPR2),
                     ('SpherocityIndex', Descriptors3D.SpherocityIndex)]:
        try: d[name] = fn(mol_h)
        except: d[name] = np.nan
    return d

# 3D for all compounds
print("\n3D conformer generation...")
shape_data = []
for _, row in df.iterrows():
    d = gen_3d(row['mol'])
    d['Identifier'] = row['Identifier']
    d['is_active'] = row['is_active']
    shape_data.append(d)
shape_df = pd.DataFrame(shape_data)

print("\n3D shape descriptors (active vs inactive):")
for col in ['Asphericity', 'Eccentricity', 'NPR1', 'NPR2', 'SpherocityIndex']:
    av = shape_df[shape_df['is_active']==1][col].dropna()
    iv = shape_df[shape_df['is_active']==0][col].dropna()
    if len(av)>0 and len(iv)>0:
        print(f"  {col}: active={av.mean():.3f}±{av.std():.3f}, inactive={iv.mean():.3f}±{iv.std():.3f}")
shape_df.to_csv(os.path.join(OUT_3D, 'shape_analysis.csv'), index=False)

# Stereo analysis
print("\nStereochemistry vs Activity:")
stereo_df = df[['Identifier', 'is_active', 'stereo_stereo_status', 'stereo_n_stereo_centers',
                 'stereo_n_specified', 'stereo_n_unspecified']].copy()
print(stereo_df.groupby(['is_active', 'stereo_stereo_status']).size().unstack(fill_value=0))
stereo_df.to_csv(os.path.join(OUT_STEREO, 'stereo_analysis.csv'), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. LOO CV
ax = axes[0, 0]
names = list(cv_results.keys())
accs = [cv_results[n] for n in names]
bars = ax.bar(names, accs, color=['steelblue', 'coral', 'green'], edgecolor='black')
ax.set_ylabel('LOO Accuracy')
ax.set_title('Leave-One-Out Cross-Validation')
ax.set_ylim([0, 1.1])
for b, a in zip(bars, accs):
    ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.02, f'{a:.3f}', ha='center', fontweight='bold')

# 2. Descriptor importance
ax = axes[0, 1]
imp_sorted = sorted(zip(descriptor_cols, desc_importances), key=lambda x: x[1], reverse=True)[:8]
nms, imps = zip(*imp_sorted)
ax.barh(range(len(nms)), imps, color='steelblue', edgecolor='black')
ax.set_yticks(range(len(nms))); ax.set_yticklabels(nms)
ax.set_xlabel('Gini Importance'); ax.set_title('Descriptor Importance')

# 3. Feature ablation
ax = axes[0, 2]
a_df = ablation_df.sort_values('accuracy_drop', ascending=True)
ax.barh(range(len(a_df)), a_df['accuracy_drop'], color='coral', edgecolor='black')
ax.set_yticks(range(len(a_df))); ax.set_yticklabels(a_df['feature'])
ax.set_xlabel('Accuracy Drop'); ax.set_title('Feature Ablation')

# 4. TAF enrichment
ax = axes[1, 0]
tp = taf_df.sort_values('Enrichment', ascending=True)
colors_map = {'Strong': 'red', 'Moderate': 'orange', 'Weak': 'gray'}
ax.barh(range(len(tp)), tp['Enrichment'], color=[colors_map.get(e,'gray') for e in tp['Evidence']], edgecolor='black')
ax.set_yticks(range(len(tp))); ax.set_yticklabels([t.split(':')[0] for t in tp['TAF']], fontsize=8)
ax.axvline(x=1, color='black', linestyle='--', linewidth=0.5)
ax.set_xlabel('Enrichment'); ax.set_title('TAF Evidence')

# 5. Stereochemistry
ax = axes[1, 1]
sc = stereo_df.groupby(['is_active', 'stereo_stereo_status']).size().unstack(fill_value=0)
sc.plot(kind='bar', ax=ax, colormap='Set2')
ax.set_xlabel('Is Active'); ax.set_ylabel('Count'); ax.set_title('Stereochemistry vs Activity')
ax.legend(fontsize=7, title='Status')

# 6. 3D shape
ax = axes[1, 2]
for col in ['Asphericity', 'NPR1']:
    av = shape_df[shape_df['is_active']==1][col].dropna()
    iv = shape_df[shape_df['is_active']==0][col].dropna()
    if len(av)>0 and len(iv)>0:
        ax.scatter([col]*len(av), av, c='red', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.scatter([col]*len(iv), iv, c='gray', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.set_ylabel('Value'); ax.set_title('3D Shape: Active (red) vs Inactive (gray)')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '04_taf_stereo.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved figure: {FIG_DIR}/04_taf_stereo.png")

# Save counterfactual
pd.DataFrame(counterfactual_results).to_csv(os.path.join(OUT_TAF, 'counterfactual.csv'), index=False)

# TAF Transferability score formula
print("\n" + "=" * 70)
print("TAF TRANSFERABILITY DEFINITION")
print("=" * 70)
print("""
T(F) = active diverse scaffolds containing F / tested diverse scaffolds containing F

Initially prospective - becomes evidence after testing on novel scaffolds.

TAF-1 (Lactone): T(F) = not yet tested on novel scaffolds
TAF-2 (Hydroxyl): T(F) = not yet tested on novel scaffolds
TAF-3 (EWG): T(F) = not yet tested on novel scaffolds
TAF-4 (Stereo): T(F) = not yet tested on novel scaffolds
""")

print("\n=== STEPS 5-7 COMPLETE ===")
