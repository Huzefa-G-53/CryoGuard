"""
Ablation Study: Baseline vs. Pure ML vs. ML + Physics Digital Twin (F2).

Evaluates the exact impact of adding physics-informed features (dt_residual,
dt_health_index, parameter drift) on precision, recall, F1, and false alarm reduction.
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

# Dynamically ensure search roots are in sys.path
current_dir = Path(__file__).resolve().parent
search_roots = [
    current_dir.parent.parent,          # e.g., d:\CryoGuard\cryoguard
    current_dir.parent.parent.parent,   # e.g., d:\CryoGuard
    Path.cwd()
]
for r in search_roots:
    if str(r) not in sys.path:
        sys.path.insert(0, str(r))

try:
    from xgboost import XGBClassifier
    MODEL_TYPE = 'xgboost'
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    MODEL_TYPE = 'sklearn'

from cryoguard.src.physics.digital_twin import add_digital_twin_features

# Constants
TEMP_LIMIT = 8.0   # °C
HORIZONS = [15, 30, 60, 120]
RANDOM_SEED = 42

def find_dataset():
    candidates = [
        Path('cryoguard/data/brand_a_processed.csv'),
        Path('data/brand_a_processed.csv'),
        Path('cryoguard/data/combined_brands_processed.csv'),
        Path('data/combined_brands_processed.csv'),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def compute_metrics(y_true, y_pred, duration_hours):
    """Compute precision, recall, F1, and false alarms per week."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fa_per_hour = fp / duration_hours if duration_hours > 0 else 0.0
    fa_per_week = fa_per_hour * 24 * 7
    return {
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'recall': float(recall), 'precision': float(precision), 'f1': float(f1),
        'false_alarms_per_week': float(fa_per_week)
    }

def run_ablation():
    data_path = find_dataset()
    if not data_path:
        print("ERROR: No processed dataset found to benchmark.")
        return

    print("=" * 90)
    print("CRYOGUARD AI - PHYSICS DIGITAL TWIN (F2) ABLATION STUDY")
    print("=" * 90)
    print(f"Loading telemetry from: {data_path}")

    # Load 100,000 continuous time-series rows for fast and robust benchmarking
    df_raw = pd.read_csv(data_path, nrows=100000, parse_dates=['timestamp'])
    df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)
    print(f"Loaded {len(df_raw):,} time-series records.")

    # 1. Compute Physics Digital Twin features (update every 5 minutes)
    print("\n[Step 1/3] Computing Physics Digital Twin signals (residuals & parameter drift)...")
    start_dt = time.time()
    df_enhanced = add_digital_twin_features(df_raw, window_size=30, update_every=5)
    dt_duration = time.time() - start_dt
    print(f"Physics simulation completed in {dt_duration:.2f}s.")

    # Fill NaNs for physics columns
    physics_cols = ['dt_residual', 'dt_health_index', 'dt_ua_drift', 'dt_cooling_drift']
    for col in physics_cols:
        if col in df_enhanced.columns:
            df_enhanced[col] = df_enhanced[col].fillna(0.0)

    # 2. Define Features
    exclude_cols = {'timestamp', 'brand', 'unit_id'}
    for h in HORIZONS:
        exclude_cols.add(f'excursion_future_{h}m')
    for col in df_enhanced.columns:
        if col.startswith('dt_'):
            exclude_cols.add(col)

    base_features = [c for c in df_enhanced.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_enhanced[c])]
    physics_features = base_features + physics_cols

    print(f"Base Telemetry Features ({len(base_features)})")
    print(f"Physics Features Added (4): {physics_cols}")

    # 3. Train/Test Split (70/30 chronological to avoid lookahead leakage)
    split_idx = int(len(df_enhanced) * 0.70)
    train_df = df_enhanced.iloc[:split_idx].reset_index(drop=True)
    test_df = df_enhanced.iloc[split_idx:].reset_index(drop=True)

    test_duration_hours = (test_df['timestamp'].max() - test_df['timestamp'].min()).total_seconds() / 3600

    results = []

    print("\n[Step 2/3] Training and Evaluating across Horizons...")
    for horizon in HORIZONS:
        target_col = f'excursion_future_{horizon}m'
        y_train = train_df[target_col]
        y_true = test_df[target_col]

        # Calculate class weight to handle imbalance
        pos_count = max(1, int(y_train.sum()))
        neg_count = int((y_train == 0).sum())
        pos_weight = min(20.0, neg_count / pos_count)

        # A. Baseline (Current Temp > 8°C)
        y_pred_base = (test_df['temp_fridge_c'] > TEMP_LIMIT).astype(int)
        base_res = compute_metrics(y_true, y_pred_base, test_duration_hours)
        base_res.update({'horizon': horizon, 'model': 'Baseline (Threshold >8C)'})
        results.append(base_res)

        # B. Pure ML (Telemetry Only)
        if MODEL_TYPE == 'xgboost':
            m_no_phys = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8, scale_pos_weight=pos_weight,
                random_state=RANDOM_SEED, n_jobs=-1, eval_metric='logloss'
            )
        else:
            m_no_phys = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_SEED)

        m_no_phys.fit(train_df[base_features], y_train)
        y_pred_no_phys = m_no_phys.predict(test_df[base_features])
        no_phys_res = compute_metrics(y_true, y_pred_no_phys, test_duration_hours)
        no_phys_res.update({'horizon': horizon, 'model': 'ML (Telemetry Only)'})
        results.append(no_phys_res)

        # C. ML + Physics Digital Twin (F2)
        if MODEL_TYPE == 'xgboost':
            m_phys = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8, scale_pos_weight=pos_weight,
                random_state=RANDOM_SEED, n_jobs=-1, eval_metric='logloss'
            )
        else:
            m_phys = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_SEED)

        m_phys.fit(train_df[physics_features], y_train)
        y_pred_phys = m_phys.predict(test_df[physics_features])
        phys_res = compute_metrics(y_true, y_pred_phys, test_duration_hours)
        phys_res.update({'horizon': horizon, 'model': 'ML + Physics Digital Twin (CryoGuard)'})
        results.append(phys_res)

        print(f"\nHorizon {horizon}m:")
        print(f"  Baseline:     Prec={base_res['precision']*100:.1f}% | Rec={base_res['recall']*100:.1f}% | F1={base_res['f1']:.3f} | False Alarms/Wk={base_res['false_alarms_per_week']:.0f}")
        print(f"  ML Only:      Prec={no_phys_res['precision']*100:.1f}% | Rec={no_phys_res['recall']*100:.1f}% | F1={no_phys_res['f1']:.3f} | False Alarms/Wk={no_phys_res['false_alarms_per_week']:.0f}")
        print(f"  ML + Physics: Prec={phys_res['precision']*100:.1f}% | Rec={phys_res['recall']*100:.1f}% | F1={phys_res['f1']:.3f} | False Alarms/Wk={phys_res['false_alarms_per_week']:.0f}")

    # 4. Save to CSV
    results_df = pd.DataFrame(results)
    out_dir = Path('cryoguard/data')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'physics_ablation_comparison.csv'
    results_df.to_csv(out_csv, index=False)
    print(f"\n[Step 3/3] Detailed ablation results saved to: {out_csv}")

    # Print clean summary
    print("\n" + "=" * 95)
    print("FINAL ABLATION TABLE: THE EXACT IMPACT OF ADDING THE PHYSICS DIGITAL TWIN")
    print("=" * 95)
    header = f"{'Horizon':<10} {'Model Configuration':<38} {'Precision':<12} {'Recall':<12} {'F1-Score':<10} {'False Alarms/Wk':<16}"
    print(header)
    print("-" * 95)
    for r in results:
        print(f"{r['horizon']:<10} {r['model']:<38} {r['precision']*100:<11.1f}% {r['recall']*100:<11.1f}% {r['f1']:<10.3f} {r['false_alarms_per_week']:<16.1f}")
    print("=" * 95)

if __name__ == "__main__":
    run_ablation()
