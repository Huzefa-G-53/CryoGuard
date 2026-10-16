"""
Feature Engineering Module for CryoGuard AI.

Unifies raw telemetry features, derived rate-of-change signals,
and F2 Digital Twin physics features (residuals, parameter drift).
"""

from typing import List, Optional
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project roots are in sys.path
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
    from cryoguard.src.physics.digital_twin import add_digital_twin_features
except ImportError:
    from src.physics.digital_twin import add_digital_twin_features

# Standard candidate telemetry features
BASE_TELEMETRY_FEATURES: List[str] = [
    'temp_fridge_c', 'temp_freezer_c', 'ambient_temp_c',
    'evaporator_temp_c', 'condenser_temp_c', 'compressor_state',
    'compressor_current_a', 'duty_cycle_15m', 'defrost_state',
    'defrost_heater_current_a', 'fan_rpm', 'door_open',
    'door_open_duration', 'power_w', 'energy_kwh',
    'rate_of_change'
]

PHYSICS_FEATURES: List[str] = [
    'dt_residual',
    'dt_health_index',
    'dt_ua_drift',
    'dt_cooling_drift',
    'dt_load_drift',
    'dt_door_effect_drift',
    'dt_predicted_delta'
]


def extract_available_features(df: pd.DataFrame, include_physics: bool = True) -> List[str]:
    """
    Return all valid numeric feature column names present in the dataframe.
    """
    candidate_features = list(BASE_TELEMETRY_FEATURES)
    if include_physics:
        candidate_features.extend(PHYSICS_FEATURES)

    # Exclude targets, identifiers, and timestamps
    exclude = {'timestamp', 'brand', 'unit_id'}
    for col in df.columns:
        if col.startswith('excursion_future_'):
            exclude.add(col)

    available = []
    for col in candidate_features:
        if col in df.columns and col not in exclude:
            if pd.api.types.is_numeric_dtype(df[col]):
                available.append(col)

    return available


def prepare_features(df: pd.DataFrame, include_physics: bool = True, window_size: int = 30) -> pd.DataFrame:
    """
    Prepare a dataframe with full engineered features.

    1. Computes rate_of_change if missing.
    2. Computes F2 Digital Twin physics features (residuals & parameter drift) if requested.
    3. Handles NaN values gracefully.

    Args:
        df: Input raw or semi-processed DataFrame
        include_physics: Whether to run the physics digital twin simulation
        window_size: Rolling window size for the digital twin

    Returns:
        DataFrame enriched with features.
    """
    enhanced = df.copy()

    # 1. Rate of change
    if 'rate_of_change' not in enhanced.columns and 'temp_fridge_c' in enhanced.columns:
        enhanced['rate_of_change'] = enhanced['temp_fridge_c'].diff().fillna(0.0)

    # 2. Physics Digital Twin
    if include_physics:
        required_physics_cols = ['temp_fridge_c', 'ambient_temp_c', 'door_open', 'compressor_state']
        if all(col in enhanced.columns for col in required_physics_cols):
            # Check if physics features already computed
            if 'dt_residual' not in enhanced.columns:
                enhanced = add_digital_twin_features(enhanced, window_size=window_size)
                # Fill initial window NaNs with 0 for residual and health index
                for col in PHYSICS_FEATURES:
                    if col in enhanced.columns:
                        enhanced[col] = enhanced[col].fillna(0.0)

    return enhanced


if __name__ == "__main__":
    print("CryoGuard AI Feature Engineering Module")
    print(f"Base Telemetry Features: {len(BASE_TELEMETRY_FEATURES)}")
    print(f"Physics Digital Twin Features: {PHYSICS_FEATURES}")
