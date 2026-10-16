import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
import time
import os

# Try to import xgboost, if not available use sklearn's GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    model_type = 'xgboost'
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    model_type = 'sklearn'
    print("XGBoost not installed, using sklearn's GradientBoostingClassifier")

# Constants
TEMP_LIMIT = 8.0  # °C, threshold for excursion
HORIZONS = [15, 30, 60, 120]  # minutes
RANDOM_SEED = 42
TEST_SIZE = 0.5  # 50% for testing, 50% for training

def load_data(filepath):
    """Load the processed data."""
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    # Sort by timestamp just in case
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def split_data(df, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    """Split data into train and test sets randomly (not time-series aware for quick demo)."""
    print(f"Splitting data into {100*(1-test_size)}% train and {100*test_size}% test (random split)")
    # We'll split the indices randomly
    train_indices, test_indices = train_test_split(
        df.index, test_size=test_size, random_state=random_state, stratify=df[f'excursion_future_{HORIZONS[0]}m']
    )
    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)
    print(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")
    return train_df, test_df

def get_features(df):
    """Return feature columns (excluding target and timestamp), only numeric ones."""
    # Exclude target columns and timestamp
    exclude = ['timestamp'] + [f'excursion_future_{h}m' for h in HORIZONS]
    # Get all remaining columns
    candidate_cols = [col for col in df.columns if col not in exclude]
    # Filter to only numeric columns (int, float, bool)
    numeric_cols = []
    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        # Optionally, we could convert bool to int, but they're already numeric in most cases
    return numeric_cols

def evaluate_baseline(test_df, horizon):
    """Evaluate the simple threshold baseline for a given horizon."""
    # Baseline: if current temperature > TEMP_LIMIT, predict excursion will happen in the next `horizon` minutes
    y_pred = (test_df['temp_fridge_c'] > TEMP_LIMIT).astype(int)
    y_true = test_df[f'excursion_future_{horizon}m']

    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate false alerts per hour in the test set
    # We need the duration of the test set in hours
    duration_hours = (test_df['timestamp'].max() - test_df['timestamp'].min()).total_seconds() / 3600
    false_alarms_per_hour = fp / duration_hours if duration_hours > 0 else 0.0
    false_alarms_per_week = false_alarms_per_hour * 24 * 7

    return {
        'horizon': horizon,
        'model': 'Baseline (Threshold)',
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'recall': recall, 'precision': precision, 'f1': f1,
        'false_alarms_per_hour': false_alarms_per_hour,
        'false_alarms_per_week': false_alarms_per_week
    }

def train_and_evaluate_model(train_df, test_df, horizon):
    """Train a Gradient Boosting Classifier and evaluate on test set for a given horizon."""
    feature_cols = get_features(train_df)
    print(f"  Using {len(feature_cols)} numeric features for training")
    X_train = train_df[feature_cols]
    y_train = train_df[f'excursion_future_{horizon}m']
    X_test = test_df[feature_cols]
    y_test = test_df[f'excursion_future_{horizon}m']

    print(f"  Training model for {horizon}-minute horizon...")
    start_time = time.time()

    if model_type == 'xgboost':
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_SEED
        )

    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Predict probabilities and convert to binary with threshold 0.5
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate false alerts per hour in the test set
    duration_hours = (test_df['timestamp'].max() - test_df['timestamp'].min()).total_seconds() / 3600
    false_alarms_per_hour = fp / duration_hours if duration_hours > 0 else 0.0
    false_alarms_per_week = false_alarms_per_hour * 24 * 7

    return {
        'horizon': horizon,
        'model': f'{model_type.capitalize()} Classifier',
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'recall': recall, 'precision': precision, 'f1': f1,
        'false_alarms_per_hour': false_alarms_per_hour,
        'false_alarms_per_week': false_alarms_per_week,
        'training_time_sec': training_time
    }

def main():
    # Resolve combined brands data path
    candidates = [
        os.path.join('cryoguard', 'data', 'combined_brands_processed.csv'),
        os.path.join('data', 'combined_brands_processed.csv'),
        os.path.join('processed_data', 'combined_brands_processed.csv'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'combined_brands_processed.csv'),
    ]
    data_path = None
    for p in candidates:
        if os.path.exists(p):
            data_path = p
            break

    if data_path is None:
        print(f"Error: combined_brands_processed.csv not found in candidate paths: {candidates}")
        return

    df = load_data(data_path)

    # Split the data
    train_df, test_df = split_data(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    # We'll collect results for each horizon and each model
    results = []

    for horizon in HORIZONS:
        print(f"\n=== Evaluating for {horizon}-minute horizon ===")

        # Evaluate baseline
        baseline_result = evaluate_baseline(test_df, horizon)
        results.append(baseline_result)
        print(f"Baseline (Threshold): Recall={baseline_result['recall']:.4f}, "
              f"F1={baseline_result['f1']:.4f}, "
              f"False Alarms/Week={baseline_result['false_alarms_per_week']:.2f}")

        # Train and evaluate model
        model_result = train_and_evaluate_model(train_df, test_df, horizon)
        results.append(model_result)
        print(f"{model_result['model']}: Recall={model_result['recall']:.4f}, "
              f"F1={model_result['f1']:.4f}, "
              f"False Alarms/Week={model_result['false_alarms_per_week']:.2f}, "
              f"Training Time={model_result['training_time_sec']:.2f}s")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(f"{'Horizon':<10} {'Model':<25} {'Recall':<8} {'F1':<8} {'False Alarms/Week':<18} {'Notes'}")
    print("-"*80)
    for res in results:
        notes = ""
        if res['model'] == 'Baseline (Threshold)':
            notes = "(Current temp > 8°C)"
        else:
            notes = f"(Trained in {res.get('training_time_sec', 0):.1f}s)"
        print(f"{res['horizon']:<10} {res['model']:<25} {res['recall']:<8.4f} {res['f1']:<8.4f} "
              f"{res['false_alarms_per_week']:<18.2f} {notes}")

    # Save results to CSV for later use
    results_df = pd.DataFrame(results)
    out_dir = os.path.dirname(data_path) if data_path else 'cryoguard/data'
    out_file = os.path.join(out_dir, 'baseline_vs_model_comparison.csv')
    results_df.to_csv(out_file, index=False)
    print(f"\nDetailed results saved to {out_file}")

if __name__ == "__main__":
    main()