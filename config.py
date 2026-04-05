from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "city_day.csv"
MODEL_FILE = PROJECT_DIR / "xgboost_aqi_model.pkl"
PLOT_FILE = PROJECT_DIR / "actual_vs_predicted_aqi.png"

FEATURE_COLUMNS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
TARGET_COLUMN = "AQI"
TEST_SIZE = 0.2
RANDOM_STATE = 42

SAMPLE_INPUT = [[80, 120, 30, 15, 1.2, 25]]
