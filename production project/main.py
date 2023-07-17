from datetime import datetime
import pandas as pd
import logging
from collections import Counter
import numpy as np
from pycaret.classification import ClassificationExperiment
from pathlib import Path
from rich.logging import RichHandler
import model_setup
import mlflow
from api_configuration import start_server


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler()],
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


# drop customer_ID and S_2 from train and test data


def drop_customer_id_s2(train_rndm_sample, test_rndm_sample):
    """
    Drops customer_ID and S_2 columns from train and test data.

    Args:
        train_rndm_sample (pd.DataFrame): The train data to be sampled.
        test_rndm_sample (pd.DataFrame): The test data to be sampled.

    Returns:
        pd.DataFrame: The train data with customer_ID and S_2 columns dropped.
        pd.DataFrame: The test data with customer_ID and S_2 columns dropped.
    """
    logging.info("Dropping customer_ID and S_2 columns from train and test data")
    train_rndm_sample = train_rndm_sample.drop(["customer_ID", "S_2"], axis=1)
    test_rndm_sample = test_rndm_sample.drop(["customer_ID", "S_2"], axis=1)
    logging.info("Dropped customer_ID and S_2 columns from train and test data")
    return train_rndm_sample, test_rndm_sample


if __name__ == "__main__":
    path = "data/train_data.ftr"
    train_data = load_data(path)
    train_data = sort_train_data(train_data)
    train_rndm_sample, test_rndm_sample = pick_random_sample(
        train_data, n_train=2000, n_test=1000
    )

    train_rndm_sample, test_rndm_sample = drop_customer_id_s2(
        train_rndm_sample, test_rndm_sample
    )
    # setup the classification experiment
    cls_model = model_setup.classification_model_setup(
        train_rndm_sample,
        target="target",
        normalize=True,
        fold=5,
        feature_selection=True,
    )
    compare_models_all = model_setup.compare_models(cls_model, n_select=5)

    # save the top n best models
    best_models = model_setup.save_best_models(
        cls_model, compare_models_all, n_select=3
    )

    # tune the top n best models
    tuned_best_models = model_setup.tune_best_models(cls_model, best_models, n_select=3)

    # end previous mlflow run
    mlflow.end_run()

    # use the tuned best models to get predictions

    pred_scores_df = model_setup.get_base_predictions(
        cls_model, tuned_best_models, test_rndm_sample
    )

    # start the API server
    start_server()
