"""Small script to generate controlled drift scenarios for testing.
Usage:
    python tools/drift_simulate.py --scenario income_drop --out data/drift_sim.csv
"""
import argparse
import pandas as pd
import numpy as np


def simulate_income_drop(baseline_path, out_path, scale=0.7, shift=-10000, n=None):
    df = pd.read_csv(baseline_path)
    if n:
        df = df.sample(n=n, replace=True, random_state=99)
    np.random.seed(42)
    df = df.copy()
    # scale incomes and shift
    df['Income'] = (df['Income'] * scale).astype(int) + int(shift)
    df.to_csv(out_path, index=False)
    print(f"Saved drifted file to: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', choices=['income_drop','credit_shift'], default='income_drop')
    parser.add_argument('--baseline', default='data/baseline.csv')
    parser.add_argument('--out', default='data/drift_sim.csv')
    parser.add_argument('--n', type=int, default=5000)
    args = parser.parse_args()

    if args.scenario == 'income_drop':
        simulate_income_drop(args.baseline, args.out, scale=0.6, shift=-5000, n=args.n)
    else:
        print('Scenario not implemented')
