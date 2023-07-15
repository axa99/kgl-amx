from datetime import datetime
import pandas as pd
import logging
from collections import Counter
import numpy as np
from pycaret.classification import ClassificationExperiment
from pathlib import Path
from rich import console


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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


def pick_random_sample(train_data, n_train=2000, n_test=1000):
    """
    Picks a random sample of customer_IDs from train_data.

    Args:
        train_data (pd.DataFrame): The train data to be sampled.
        n_train (int): The number of customer_IDs to be sampled for training data.
        n_test (int): The number of customer_IDs to be sampled for test data.

    Returns:
        pd.DataFrame: The sampled train data.
        pd.DataFrame: The sampled test data.
    """
    logging.info(f"Picking random sample of {n_train} customer_IDs from train_data")
    np.random.seed(42)
    customer_ID_rand = np.random.choice(
        train_data["customer_ID"], n_train, replace=False
    )
    train_rndm_sample = train_data[train_data["customer_ID"].isin(customer_ID_rand)]

    # # customer_IDs that are not in the random sample for the training data
    customer_ID_not_train = list(
        set(train_data["customer_ID"]).difference(set(customer_ID_rand))
    )

    # # pick random sample of customer_IDs from customer_ID_not_train
    logging.info(
        f"Picking random sample of {n_test} customer_IDs from customer_ID_not_train"
    )
    customer_ID_test_rnd = np.random.choice(
        customer_ID_not_train, n_test, replace=False
    )

    # # test data based on the random sample of customer_IDs
    test_rndm_sample = train_data[
        train_data["customer_ID"].isin(customer_ID_test_rnd)
    ].sort_values(by=["customer_ID", "S_2"])

    # ration of data in train versus test
    logging.info(
        f"Ratio of data in train versus test sample is {round(len(train_rndm_sample)/len(test_rndm_sample), 2)}"
    )
    logging.info(f"Train data shape: {train_rndm_sample.shape}")
    logging.info(f"Test data shape: {test_rndm_sample.shape}")

    return train_rndm_sample, test_rndm_sample


if __name__ == "__main__":
    path = "data/train_data.ftr"
    train_data = load_data(path)
    train_data = sort_train_data(train_data)
    train_rndm_sample, test_rndm_sample = pick_random_sample(
        train_data, n_train=2000, n_test=1000
    )
