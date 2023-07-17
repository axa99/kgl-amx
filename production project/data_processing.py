import pandas as pd
import logging


def load_data(path):
    """
    Loads data from a given path.

    Args:
        path (str): The path to the data.

    Returns:
        pd.DataFrame: The loaded data.
    """
    logging.info(f"Loading data from {path}")
    data = pd.read_feather(path)
    logging.info(f"Data loaded from {path} with shape {data.shape}")
    return data


def sort_train_data(train_data):
    """
    Sorts train_data by customer_ID and S_2 columns.

    Args:
        train_data (pd.DataFrame): The train data to be sorted.

    Returns:
        pd.DataFrame: The sorted train data.
    """
    logging.info("Sorting train data by customer_ID and S_2 columns")
    train_data["S_2"] = pd.to_datetime(train_data["S_2"])
    sorted_train_data = train_data.sort_values(by=["customer_ID", "S_2"])
    logging.info("Train data sorted by customer_ID and S_2 columns")
    return sorted_train_data


if __name__ == "__main__":
    print("This is the data-processing.py file")
