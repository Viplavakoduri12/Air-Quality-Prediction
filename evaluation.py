import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from config import PLOT_FILE


def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    metrics = {
        "mse": mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }
    return predictions, metrics


def save_evaluation_plot(y_test, predictions, file_path=PLOT_FILE) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, predictions)
    ax.set_xlabel("Actual AQI")
    ax.set_ylabel("Predicted AQI")
    ax.set_title("Actual vs Predicted AQI")
    fig.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)
