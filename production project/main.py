from datetime import datetime
import pandas as pd
import logging
from collections import Counter
import numpy as np
from pycaret.classification import ClassificationExperiment
from pathlib import Path
from rich import print


# Load data
def load_data(path):
    """
    Loads data from a given path.

    Args:
        path (str): The path to the data.

    Returns:
        pd.DataFrame: The loaded data.
    """
    data = pd.read_feather(path)
    print(f"Data loaded from {path} with shape {data.shape}")
    return data


def sort_train_data(train_data):
    """
    Sorts train_data by customer_ID and S_2 columns.

    Args:
        train_data (pd.DataFrame): The train data to be sorted.

    Returns:
        pd.DataFrame: The sorted train data.
    """
    train_data["S_2"] = pd.to_datetime(train_data["S_2"])
    sorted_train_data = train_data.sort_values(by=["customer_ID", "S_2"])

    print("Train data sorted by customer_ID and S_2 columns")
    return sorted_train_data


# pick random sample of customer_IDs from train_data
def pick_random_sample(train_data, n_train=2000, n_test=1000):
    """
    Picks a random sample of customer_IDs from train_data.

    Args:
        train_data (pd.DataFrame): The train data to be sampled.
        n (int): The number of customer_IDs to be sampled.

    Returns:
        np.array: The sampled customer_IDs.
    """
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
    customer_ID_test_rnd = np.random.choice(
        customer_ID_not_train, n_test, replace=False
    )

    # # test data based on the random sample of customer_IDs
    test_rndm_sample = train_data[
        train_data["customer_ID"].isin(customer_ID_test_rnd)
    ].sort_values(by=["customer_ID", "S_2"])

    # ration of data in train versus test
    print(
        f"Ratio of data in train versus test sample is {round(len(train_rndm_sample)/len(test_rndm_sample), 2)}\
          \nTrain data shape: {train_rndm_sample.shape}\nTest data shape: {test_rndm_sample.shape}"
    )

    return train_rndm_sample, test_rndm_sample


# # %%
# # drop customer_ID and S_2 from train and test data
# train_rndm_sample.drop(["customer_ID", "S_2"], axis=1, inplace=True)
# test_rndm_sample.drop(["customer_ID", "S_2"], axis=1, inplace=True)

# # %%
# # calculate the percentage of training data sample that is 1 and 0
# print(train_rndm_sample["target"].value_counts(normalize=True))
# train_rndm_sample["target"].value_counts().plot(kind="bar", title="Target Distribution")

# # %%
# s = ClassificationExperiment()
# s.setup(
#     data=train_rndm_sample,
#     target="target",
#     normalize=True,
#     session_id=123,
#     fix_imbalance=True,
#     fold=5,
#     feature_selection=True,
#     remove_multicollinearity=True,
#     multicollinearity_threshold=0.9,
# )

# # %%
# s.models()

# # %%
# number_of_models = len(s.models())
# compare_models_all = s.compare_models(n_select=number_of_models)

# # %%
# pred_scores_df = pd.DataFrame()
# for x in compare_models_all:
#     df_pred = s.predict_model(
#         x,
#         data=test_rndm_sample,
#     )
#     x = s.pull()
#     pred_scores_df = pd.concat([pred_scores_df, x])

# # %%
# display(pred_scores_df.sort_values(by=["Accuracy"], ascending=False))

# # %%
# for model in compare_models_all:
#     s.save_model(model, "../models_saved/" + model.__class__.__name__)

# # %%
# # Load saved models
# catboost = s.load_model("../models_saved/CatBoostClassifier")
# catboost

# # %%
# s.predict_model(catboost, data=test_rndm_sample)

# # %%
# tuned_cat_boost, catboost_tuner = s.tune_model(
#     catboost,
#     custom_grid={"learning_rate": [0.01, 0.03], "depth": [4, 6, 8, 10]},
#     return_tuner=True,
# )

# # %%
# pd.DataFrame(catboost_tuner.cv_results_)

# # %%
# s.save_model(tuned_cat_boost, "../models_saved/tuned_cat_boost_07_11_2023")

# # %%
# tuned_cat_boost = s.load_model("../models_saved/tuned_cat_boost_07_11_2023")

# # %%
# s.plot_model(tuned_cat_boost, plot="confusion_matrix")

# # %%
# # check the residuals of trained model**
# s.plot_model(tuned_cat_boost, plot="feature_all")

# # %%
# s.plot_model(tuned_cat_boost, plot="pipeline", plot_kwargs={"fontsize": 16})

# # %%
# s.plot_model(tuned_cat_boost, plot="boundary")

# # %%
# s.create_app(tuned_cat_boost, app_kwargs={"title": "CatBoost"})
if __name__ == "__main__":
    path = "data/train_data.ftr"
    train_data = load_data(path)
    train_data = sort_train_data(train_data)
    train_rndm_sample, test_rndm_sample = pick_random_sample(
        train_data, n_train=2000, n_test=1000
    )
