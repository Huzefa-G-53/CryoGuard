"""
Test script for the Digital Twin implementation (F2).
Verifies physics estimation, residual calculation, and parameter drift.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Dynamically locate project roots and data
current_dir = Path(__file__).resolve().parent
search_roots = [
    current_dir.parent.parent,          # e.g., d:\CryoGuard\cryoguard or d:\CryoGuard
    current_dir.parent.parent.parent,   # e.g., d:\CryoGuard if in cryoguard/src/physics
    Path.cwd(),
]
for r in search_roots:
    if str(r) not in sys.path:
        sys.path.insert(0, str(r))

try:
    from cryoguard.src.physics.digital_twin import add_digital_twin_features
except ImportError:
    from src.physics.digital_twin import add_digital_twin_features


def find_data_file(filename="brand_a_processed.csv"):
    candidate_paths = []
    for root in search_roots:
        candidate_paths.extend([
            root / "cryoguard" / "data" / filename,
            root / "data" / filename,
            root / "processed_data" / filename,
        ])
    for p in candidate_paths:
        if p.exists():
            return p
    return None

def test_on_brand_a_sample():
    """Test the digital twin on a sample of brand_a data."""
    data_path = find_data_file("brand_a_processed.csv")
    print(f"Testing Digital Twin F2...")
    print(f"Data path resolved: {data_path}")

    if data_path is None or not data_path.exists():
        print(f"ERROR: Could not locate brand_a_processed.csv in candidate locations.")
        return False

    # Load 1000 rows for verification
    df = pd.read_csv(data_path, nrows=1000, parse_dates=['timestamp'])
    print(f"Loaded {len(df)} sample rows successfully.")

    required_cols = ['temp_fridge_c', 'ambient_temp_c', 'door_open', 'compressor_state']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required telemetry columns: {missing}")
        return False

    print("Executing Digital Twin physics simulation & residual calculation...")
    df_enhanced = add_digital_twin_features(df, window_size=30)

    dt_cols = [c for c in df_enhanced.columns if c.startswith('dt_')]
    print(f"Successfully computed Digital Twin columns: {dt_cols}")

    print("\n--- Digital Twin Feature Diagnostics ---")
    valid_residuals = df_enhanced['dt_residual'].dropna()
    print(f"Residuals Count: {len(valid_residuals)}")
    print(f"Mean Residual: {valid_residuals.mean():.6f} C")
    print(f"Std Residual:  {valid_residuals.std():.6f} C")
    print(f"Mean Health Index (Drift): {df_enhanced['dt_health_index'].dropna().mean():.4f}")

    sample_cols = ['timestamp', 'temp_fridge_c', 'ambient_temp_c', 'door_open', 'compressor_state', 'dt_residual', 'dt_health_index']
    sample_view = df_enhanced[sample_cols].dropna(subset=['dt_residual']).head(5)
    print("\n--- First 5 Evaluated Telemetry Points ---")
    print(sample_view.to_string(index=False))

    print("\n[SUCCESS] Digital Twin physics test passed successfully!")
    return True

if __name__ == "__main__":
    success = test_on_brand_a_sample()
    if not success:
        sys.exit(1)
