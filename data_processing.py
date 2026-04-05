from pathlib import Path

import pandas as pd

from config import DATA_FILE, FEATURE_COLUMNS, TARGET_COLUMN


def load_raw_dataset(file_path=DATA_FILE) -> pd.DataFrame:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_csv(file_path)


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()

    for column in FEATURE_COLUMNS:
        cleaned[column] = cleaned[column].interpolate()

    cleaned = cleaned.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    return cleaned


def load_prepared_dataset(file_path=DATA_FILE) -> pd.DataFrame:
    raw_data = load_raw_dataset(file_path)
    return clean_dataset(raw_data)


def get_features_and_target(data: pd.DataFrame):
    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    return x, y
