import argparse

from data_processing import load_raw_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the AQI dataset.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Optional path to the dataset CSV file.",
    )
    args = parser.parse_args()

    data = load_raw_dataset(args.dataset) if args.dataset else load_raw_dataset()

    print("First 5 rows:")
    print(data.head())
    print()

    print("Dataset information:")
    data.info()
    print()

    print("Missing values:")
    print(data.isnull().sum())


if __name__ == "__main__":
    main()
