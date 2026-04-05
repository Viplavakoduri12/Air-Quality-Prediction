import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from config import MODEL_FILE, RANDOM_STATE, TEST_SIZE


def split_dataset(x, y):
    return train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def train_aqi_model(x_train, y_train) -> XGBRegressor:
    model = XGBRegressor(random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    return model


def save_model(model, file_path=MODEL_FILE) -> None:
    joblib.dump(model, file_path)
