"""
Digital Twin (Grey-Box) Model for Refrigeration Unit Health Monitoring.

Implements an RC thermal model with parameter tracking
to compute residuals and health indices for fault detection and early warning.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.linear_model import LinearRegression


class DigitalTwin:
    """
    A grey-box model of a refrigeration unit based on the energy balance equation:

    C * dT/dt = UA * (T_amb - T) + P_door(t) + P_load - P_cool * eta * u(t)

    where:
        C: thermal capacitance [J/°C or equivalent]
        UA: heat leak coefficient [W/°C]
        T_amb: ambient temperature [°C]
        T: refrigerator temperature [°C]
        P_door(t): heat power from door openings [W]
        P_load: constant heat load from contents and internal lighting [W]
        P_cool: cooling power when compressor is on [W]
        eta: compressor efficiency [0-1, degrades over time]
        u(t): compressor ON/OFF state (1 if on, 0 if off)

    Discretized with time step Δt:
        ΔT = T_{k+1} - T_k = a1*(T_amb - T) + a2 + a3*door_open + a4*u

    where:
        a1 = (Δt/C) * UA                  (heat leak rate)
        a2 = (Δt/C) * P_load              (base internal load)
        a3 = (Δt/C) * P_door_per_minute   (door opening effect)
        a4 = -(Δt/C) * P_cool * eta       (cooling rate, typically negative)

    The health index is based on deviation of estimated parameters from nominal:
        - Increasing a1 -> failing door seal (excessive heat leak)
        - Increasing a4 (less negative) -> compressor degradation or refrigerant loss
    """

    def __init__(self, window_size: int = 60):
        """
        Initialize the Digital Twin.

        Args:
            window_size: number of past minutes to use for parameter estimation (default 60)
        """
        self.window_size = window_size
        self.coefficients_ = None  # [a1, a2, a3, a4]
        self.is_fitted_ = False
        self.nominal_coefficients_ = None

    def _prepare_features(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """Prepare feature vector for prediction at index idx."""
        T = float(df.loc[idx, 'temp_fridge_c'])
        T_amb = float(df.loc[idx, 'ambient_temp_c'])
        door_open = float(df.loc[idx, 'door_open'])
        u = float(df.loc[idx, 'compressor_state'])

        return np.array([[T_amb - T, 1.0, door_open, u]], dtype=np.float64)

    def fit_window(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> Tuple[np.ndarray, float]:
        """
        Fit model parameters using a historical window [start_idx, end_idx].
        Uses vectorized array operations for speed.
        """
        if end_idx - start_idx < 4:
            raise ValueError("Window size must be at least 4 to estimate 4 parameters")

        # Vectorized slice
        window = df.iloc[start_idx:end_idx + 1]
        T = window['temp_fridge_c'].to_numpy(dtype=np.float64)
        T_amb = window['ambient_temp_c'].to_numpy(dtype=np.float64)
        door_open = window['door_open'].to_numpy(dtype=np.float64)
        u = window['compressor_state'].to_numpy(dtype=np.float64)

        # X from start to end-1, y as delta to next step
        X = np.column_stack([
            T_amb[:-1] - T[:-1],
            np.ones(len(T) - 1, dtype=np.float64),
            door_open[:-1],
            u[:-1]
        ])
        y = T[1:] - T[:-1]

        # Fast least squares solve
        coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        self.coefficients_ = coef
        self.is_fitted_ = True

        # R-squared calculation
        y_pred = X @ coef
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        score = 1.0 - (np.sum((y - y_pred) ** 2) / (ss_tot + 1e-12)) if ss_tot > 1e-12 else 0.0

        return self.coefficients_, float(score)

    def predict_delta(self, df: pd.DataFrame, idx: int) -> float:
        """Predict the temperature change from idx to idx+1."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction.")

        features = self._prepare_features(df, idx)
        return float(self.coefficients_.dot(features[0]))

    def compute_residual(self, df: pd.DataFrame, idx: int) -> float:
        """Compute residual: actual delta T - predicted delta T."""
        T_next = float(df.loc[idx + 1, 'temp_fridge_c'])
        T_curr = float(df.loc[idx, 'temp_fridge_c'])
        actual_delta = T_next - T_curr
        predicted_delta = self.predict_delta(df, idx)
        return float(actual_delta - predicted_delta)

    def compute_health_index(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Compute health indicators based on parameter deviation from nominal."""
        if not self.is_fitted_:
            return {
                'parameter_drift': 0.0,
                'ua_drift': 0.0,
                'cooling_drift': 0.0,
                'load_drift': 0.0,
                'door_effect_drift': 0.0
            }

        if self.nominal_coefficients_ is None:
            self.nominal_coefficients_ = self.coefficients_.copy()

        coeff_diff = self.coefficients_ - self.nominal_coefficients_
        norm_nominal = np.linalg.norm(self.nominal_coefficients_)
        if norm_nominal < 1e-8:
            norm_nominal = 1.0

        parameter_drift = float(np.linalg.norm(coeff_diff) / norm_nominal)

        return {
            'parameter_drift': parameter_drift,
            'ua_drift': float(coeff_diff[0]),
            'load_drift': float(coeff_diff[1]),
            'door_effect_drift': float(coeff_diff[2]),
            'cooling_drift': float(coeff_diff[3])
        }


def add_digital_twin_features(df: pd.DataFrame, window_size: int = 60, update_every: int = 1) -> pd.DataFrame:
    """
    High-performance vector implementation to compute Digital Twin physics features.

    Computes:
        - dt_predicted_delta: physics-expected temperature delta
        - dt_actual_delta: observed temperature delta
        - dt_residual: actual - physics expected (key anomaly signal)
        - dt_health_index: normalized coefficient drift from nominal baseline
        - dt_ua_drift: heat leak parameter drift (door seal degradation)
        - dt_cooling_drift: cooling parameter drift (compressor degradation)
        - dt_load_drift: internal heat load drift
        - dt_door_effect_drift: door opening impact drift

    Args:
        df: DataFrame containing temp_fridge_c, ambient_temp_c, door_open, compressor_state
        window_size: rolling window length in minutes
        update_every: parameter re-estimation interval (default 1 for full tracking)

    Returns:
        DataFrame with added dt_* feature columns.
    """
    df = df.copy()
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)

    n = len(df)
    # Pre-allocate numpy arrays for maximum performance
    predicted_delta = np.full(n, np.nan, dtype=np.float64)
    actual_delta = np.full(n, np.nan, dtype=np.float64)
    residuals = np.full(n, np.nan, dtype=np.float64)
    health_index = np.full(n, np.nan, dtype=np.float64)
    ua_drift = np.full(n, np.nan, dtype=np.float64)
    load_drift = np.full(n, np.nan, dtype=np.float64)
    door_drift = np.full(n, np.nan, dtype=np.float64)
    cooling_drift = np.full(n, np.nan, dtype=np.float64)

    min_idx = window_size
    max_idx = n - 2
    if min_idx > max_idx:
        df['dt_predicted_delta'] = predicted_delta
        df['dt_actual_delta'] = actual_delta
        df['dt_residual'] = residuals
        df['dt_health_index'] = health_index
        df['dt_ua_drift'] = ua_drift
        df['dt_load_drift'] = load_drift
        df['dt_door_effect_drift'] = door_drift
        df['dt_cooling_drift'] = cooling_drift
        return df

    # Extract numpy arrays for fast vectorized execution
    T = df['temp_fridge_c'].to_numpy(dtype=np.float64)
    T_amb = df['ambient_temp_c'].to_numpy(dtype=np.float64)
    door_open = df['door_open'].to_numpy(dtype=np.float64)
    u = df['compressor_state'].to_numpy(dtype=np.float64)

    # Precalculate actual delta
    actual_delta[:-1] = T[1:] - T[:-1]

    # Initialize Digital Twin
    twin = DigitalTwin(window_size=window_size)
    nominal_coef = None

    for idx in range(min_idx, max_idx + 1):
        start_idx = idx - window_size
        end_idx = idx - 1

        # Re-estimate parameters on window if on interval
        if (idx == min_idx) or (update_every == 1) or ((idx - min_idx) % update_every == 0):
            X_win = np.column_stack([
                T_amb[start_idx:end_idx + 1] - T[start_idx:end_idx + 1],
                np.ones(end_idx - start_idx + 1, dtype=np.float64),
                door_open[start_idx:end_idx + 1],
                u[start_idx:end_idx + 1]
            ])
            y_win = actual_delta[start_idx:end_idx + 1]

            # Fast least-squares solve
            try:
                coef, _, _, _ = np.linalg.lstsq(X_win, y_win, rcond=None)
                twin.coefficients_ = coef
                twin.is_fitted_ = True
                if nominal_coef is None:
                    nominal_coef = coef.copy()
                    twin.nominal_coefficients_ = nominal_coef
            except Exception:
                continue

        if not twin.is_fitted_:
            continue

        # Predict delta at current point
        feat = np.array([T_amb[idx] - T[idx], 1.0, door_open[idx], u[idx]], dtype=np.float64)
        pred_d = float(twin.coefficients_.dot(feat))
        act_d = actual_delta[idx]
        res = act_d - pred_d

        predicted_delta[idx] = pred_d
        residuals[idx] = res

        # Health index & drifts
        coeff_diff = twin.coefficients_ - nominal_coef
        norm_nom = np.linalg.norm(nominal_coef)
        if norm_nom < 1e-8:
            norm_nom = 1.0

        health_index[idx] = float(np.linalg.norm(coeff_diff) / norm_nom)
        ua_drift[idx] = float(coeff_diff[0])
        load_drift[idx] = float(coeff_diff[1])
        door_drift[idx] = float(coeff_diff[2])
        cooling_drift[idx] = float(coeff_diff[3])

    df['dt_predicted_delta'] = predicted_delta
    df['dt_actual_delta'] = actual_delta
    df['dt_residual'] = residuals
    df['dt_health_index'] = health_index
    df['dt_ua_drift'] = ua_drift
    df['dt_load_drift'] = load_drift
    df['dt_door_effect_drift'] = door_drift
    df['dt_cooling_drift'] = cooling_drift

    return df


if __name__ == "__main__":
    print("CryoGuard AI - F2 Digital Twin Module")
    print("Usage: from cryoguard.src.physics.digital_twin import add_digital_twin_features")
