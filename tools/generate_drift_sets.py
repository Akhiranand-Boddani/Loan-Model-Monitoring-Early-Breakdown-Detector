import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
BASELINE = DATA / 'baseline.csv'
OUT_NO_DRIFT = DATA / 'drift_no_drift.csv'
OUT_DRIFT = DATA / 'drift_drift.csv'

random_state = 42
np.random.seed(random_state)

print(f"Loading baseline from {BASELINE}")
base = pd.read_csv(BASELINE)

n = min(5000, len(base))
# Create no-drift: random sample preserving distributions
no_drift = base.sample(n=n, replace=False, random_state=random_state).reset_index(drop=True)

# Create drifted dataset by applying controlled shifts
drift = base.sample(n=n, replace=False, random_state=random_state+1).reset_index(drop=True)

# Numeric shifts: increase Income for 40% of rows, decrease CreditScore for 35% of rows
mask_inc = np.random.rand(n) < 0.4
mask_lowcs = np.random.rand(n) < 0.35

if 'Income' in drift.columns:
    drift.loc[mask_inc, 'Income'] = (drift.loc[mask_inc, 'Income'] * 1.25).round(0)

if 'CreditScore' in drift.columns:
    drift.loc[mask_lowcs, 'CreditScore'] = (drift.loc[mask_lowcs, 'CreditScore'] - 40).clip(lower=300)

# Categorical shift: increase 'Unemployed' proportion in EmploymentType
if 'EmploymentType' in drift.columns:
    # for 30% of rows, set to 'Unemployed'
    mask_unemp = np.random.rand(n) < 0.3
    drift.loc[mask_unemp, 'EmploymentType'] = 'Unemployed'

# Binary flag changes: flip HasMortgage for 20% to increase No->Yes
if 'HasMortgage' in drift.columns:
    # normalize values to Yes/No
    valmap = {'Yes': 'Yes', 'No': 'No', True: 'Yes', False: 'No'}
    drift['HasMortgage'] = drift['HasMortgage'].map(lambda x: valmap.get(x, x))
    mask_mort = np.random.rand(n) < 0.2
    # flip some No to Yes
    drift.loc[mask_mort & (drift['HasMortgage'] == 'No'), 'HasMortgage'] = 'Yes'

# Target drift: increase default rate slightly by flipping some 0->1 for high-risk group (low credit score)
if 'Default' in drift.columns and 'CreditScore' in drift.columns:
    # find low credit score rows
    low_cs = drift['CreditScore'] < 500
    # flip 5% of non-defaults in low_cs to default
    candidates = drift[low_cs & (drift['Default'] == 0)].index
    flip_n = max(1, int(0.05 * len(candidates)))
    flip_idx = np.random.choice(candidates, size=flip_n, replace=False)
    drift.loc[flip_idx, 'Default'] = 1

# Output stats
print('No-drift sample size:', len(no_drift))
print('Drift sample size:', len(drift))
print('Baseline default rate:', base['Default'].mean())
print('No-drift default rate:', no_drift['Default'].mean())
print('Drift default rate:', drift['Default'].mean())

# Save files
no_drift.to_csv(OUT_NO_DRIFT, index=False)
drift.to_csv(OUT_DRIFT, index=False)
print(f'Wrote {OUT_NO_DRIFT} and {OUT_DRIFT}')
