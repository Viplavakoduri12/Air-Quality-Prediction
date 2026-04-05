import argparse

import numpy as np

from config import MODEL_FILE, PLOT_FILE, SAMPLE_INPUT
from data_processing import get_features_and_target, load_prepared_dataset
from evaluation import evaluate_model, save_evaluation_plot
from model_training import save_model, split_dataset, train_aqi_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AQI prediction model.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Optional path to the dataset CSV file.",
    )
    args = parser.parse_args()

    data = load_prepared_dataset(args.dataset) if args.dataset else load_prepared_dataset()
    x, y = get_features_and_target(data)
    x_train, x_test, y_train, y_test = split_dataset(x, y)

    model = train_aqi_model(x_train, y_train)
    predictions, metrics = evaluate_model(model, x_test, y_test)

    sample_prediction = model.predict(np.array(SAMPLE_INPUT))[0]

    save_model(model)
    save_evaluation_plot(y_test, predictions)

    print(f"MSE: {metrics['mse']:.4f}")
    print(f"R2 Score: {metrics['r2']:.4f}")
    print(f"Sample Predicted AQI: {sample_prediction:.2f}")
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Evaluation plot saved to: {PLOT_FILE}")
    print(f"Test samples used for evaluation: {len(x_test)}")


if __name__ == "__main__":
    main()
