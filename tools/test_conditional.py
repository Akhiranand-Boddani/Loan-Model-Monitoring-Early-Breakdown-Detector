from loan_monitoring.drift.drift_metrics import compute_conditional_numeric_drift
import pandas as pd

base = pd.read_csv('data/baseline.csv')
new = pd.read_csv('data/drift_test.csv')
res = compute_conditional_numeric_drift(base, new, 'Income', 'Default')
print('Conditional numeric drift (Income):', res)
