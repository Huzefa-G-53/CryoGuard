import pandas as pd
import os
from pathlib import Path
import glob

def process_brand_folder(folder_path, output_dir=None, temp_limit=8.0):
    """
    Process a single brand/test folder.

    Args:
        folder_path (Path): Path to the folder containing the data
        output_dir (Path or str): Directory to save processed data
        temp_limit (float): Temperature threshold for excursion (°C)

    Returns:
        pd.DataFrame: Processed dataframe with features and labels
    """
    if output_dir is None:
        output_dir = Path('cryoguard/data') if Path('cryoguard/data').exists() else Path('data')
    output_dir = Path(output_dir)
    folder_path = Path(folder_path)
    # Find the main data CSV (handles different naming conventions)
    csv_files = list(folder_path.glob('fridge_data_*.csv'))
    if not csv_files:
        print(f"No data CSV found in {folder_path}")
        return None
    data_file = csv_files[0]

    # Load main data
    # print(f"  Loading main data from {data_file.name}")
    df = pd.read_csv(data_file, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Calculate rate of change for storage temperature (fridge compartment)
    df['rate_of_change'] = df['temp_fridge_c'].diff()
    df['rate_of_change'] = df['rate_of_change'].fillna(0.0)

    # Create labels for future excursion at multiple horizons
    horizons = [15, 30, 60, 120]  # minutes
    for h in horizons:
        future_temp = df['temp_fridge_c'].shift(-h)
        df[f'excursion_future_{h}m'] = (future_temp > temp_limit).astype(int)

    # Define feature columns we intend to use
    feature_cols = [
        'temp_fridge_c', 'temp_freezer_c', 'ambient_temp_c',
        'evaporator_temp_c', 'condenser_temp_c', 'compressor_state',
        'compressor_current_a', 'duty_cycle_15m', 'defrost_state',
        'defrost_heater_current_a', 'fan_rpm', 'door_open',
        'door_open_duration', 'power_w', 'energy_kwh',
        'rate_of_change', 'fault_indicator', 'degradation_indicator'
    ]

    # Filter to only columns that exist in the dataframe
    available_features = [f for f in feature_cols if f in df.columns]
    # We don't need to print warnings for each folder to avoid clutter

    return df, available_features

def main():
    # Resolve fridge-sensor-data and output locations
    candidates = [
        Path('fridge-sensor-data/data'),
        Path('../fridge-sensor-data/data'),
        Path('../../fridge-sensor-data/data')
    ]
    data_root = next((p for p in candidates if p.exists()), Path('fridge-sensor-data/data'))

    output_dir = Path('cryoguard/data')
    if not output_dir.parent.exists() and Path('data').exists():
        output_dir = Path('data')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all brand and test folders
    folders = [f for f in data_root.iterdir() if f.is_dir() and (f.name.startswith('brand_') or f.name.startswith('test_'))]
    folders.sort(key=lambda x: x.name)  # sort for consistent order

    print(f"Found {len(folders)} folders to process.")

    # Process each folder and save individual processed CSV
    all_features_set = set()
    brand_dfs = []  # for training (brands A-Q)
    test_dfs = []   # for OOD testing (test_01-10)

    for folder in folders:
        print(f"Processing {folder.name}...")
        result = process_brand_folder(folder, output_dir=output_dir)
        if result is None:
            continue
        df, features = result
        all_features_set.update(features)

        # Save individual processed data
        output_file = output_dir / f'{folder.name}_processed.csv'
        df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file.name} (shape: {df.shape})")

        # Separate brands and tests for train/test split
        if folder.name.startswith('brand_'):
            brand_dfs.append(df)
        elif folder.name.startswith('test_'):
            test_dfs.append(df)

    print(f"\nAll features encountered: {sorted(all_features_set)}")

    # Optionally create combined datasets
    if brand_dfs:
        combined_brand = pd.concat(brand_dfs, ignore_index=True)
        combined_brand_file = output_dir / 'combined_brands_processed.csv'
        combined_brand.to_csv(combined_brand_file, index=False)
        print(f"Combined brands dataset saved to {combined_brand_file.name} (shape: {combined_brand.shape})")

    if test_dfs:
        combined_test = pd.concat(test_dfs, ignore_index=True)
        combined_test_file = output_dir / 'combined_tests_processed.csv'
        combined_test.to_csv(combined_test_file, index=False)
        print(f"Combined tests dataset saved to {combined_test_file.name} (shape: {combined_test.shape})")

if __name__ == "__main__":
    main()