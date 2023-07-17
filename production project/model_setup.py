from pycaret.classification import ClassificationExperiment
import logging
import pandas as pd
import mlflow
import re


# setup the classification experiment
def classification_model_setup(
    train_rndm_sample,
    target: str = "target",
    normalize: bool = True,
    fold: int = 5,
    feature_selection: bool = True,
    experiment_name: str = "classification_experiment",
):
    logging.info("Setting up the classification experiment")
    s = ClassificationExperiment()
    s.setup(
        data=train_rndm_sample,
        target="target",
        normalize=True,
        session_id=123,
        fix_imbalance=True,
        fold=5,
        feature_selection=True,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.9,
        log_experiment=True,
        experiment_name="classification_experiment",
    )
    return s


def compare_models(s, n_select: int = 5):
    logging.info(f"Comparing {n_select} models")
    compare_models_all = s.compare_models(n_select=n_select)
    return compare_models_all


# save the top n best models
def save_best_models(s, compare_models_all, n_select: int = 3):
    logging.info(f"Saving the top {n_select} best models")
    best_models = compare_models_all[:n_select]
    for x in best_models:
        model_name = str(x).split("(")[0]
        model_name = re.sub(r"\s+|\<|\>", "_", model_name)
        logging.info(f"Saving {model_name}")
        s.save_model(x, f"production project/saved-models/{model_name}")
    return best_models


# get predictions for all models and save the results to a log file


def get_base_predictions(s, compare_models_all, test_rndm_sample):
    logging.info("Getting predictions for all models")
    with mlflow.start_run(run_name="classification_experiment"):
        for x in compare_models_all:
            model_name = str(x).split("(")[0]
            model_name = re.sub(r"\s+|\<|\>", "_", model_name)
            logging.info(f"Getting predictions for {model_name}")
            df_pred = s.predict_model(
                x,
                data=test_rndm_sample,
            )
            y_true = df_pred["target"]
            y_pred = df_pred["prediction_label"]
            accuracy = (y_true == y_pred).mean()

            mlflow.log_metric(f"{model_name}_accuracy", accuracy)
            file_path = f"prediction-outputs/{model_name}_predictions.csv"
            df_pred.to_csv(file_path, index=False)
            mlflow.log_artifact(file_path)


if __name__ == "__main__":
    print("This is the model_setup.py file")
