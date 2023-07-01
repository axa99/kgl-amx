import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, log_loss
from keras.callbacks import ModelCheckpoint
from joblib import Parallel, delayed

import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import pad_sequences
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.impute import KNNImputer

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import gc

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def load_data(file_path):
    print("Loading data...")
    return pd.read_feather(file_path)


def handle_missing_data(df):
    print("Handling missing data with KNN imputer...")

    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Identify numerical columns with missing data
    cols_with_missing = df[numerical_cols].isnull().any()
    numerical_cols_with_missing = cols_with_missing[
        cols_with_missing == True
    ].index.tolist()

    imputer = KNNImputer(n_neighbors=5)

    print("Applying KNN imputer on numerical columns with missing data...")
    Parallel(n_jobs=-1)(
        delayed(impute_column)(df, col) for col in numerical_cols_with_missing
    )

    # Standardize numerical columns with missing data
    print("Standardizing numerical columns with missing data...")
    scaler = StandardScaler()
    df[numerical_cols_with_missing] = scaler.fit_transform(
        df[numerical_cols_with_missing]
    )

    return df


def impute_column(df, col):
    imputer = KNNImputer(n_neighbors=5)
    df[col] = imputer.fit_transform(df[[col]])
    return df


def identify_and_process_categorical_features(df):
    print("Identifying and processing categorical features...")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [col for col in cat_cols if col not in ["customer_ID", "S_2"]]
    for col in tqdm(cat_cols):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def prepare_data(df):
    print("Preparing data...")
    df.sort_values(["customer_ID", "S_2"], inplace=True)
    grouped = df.groupby("customer_ID")
    max_sequence_length = grouped.size().max()
    feature_cols = [
        col for col in df.columns if col not in ["customer_ID", "target", "S_2"]
    ]
    return grouped, feature_cols, max_sequence_length


def prepare_sequences(grouped, feature_cols, max_sequence_length):
    X = []
    y = []
    for _, group in grouped:
        X_seq = group[feature_cols].values
        y_seq = group["target"].values[-1]
        if len(X_seq) >= max_sequence_length:
            X_seq = X_seq[-max_sequence_length:]
            X.append(X_seq)
            y.append(y_seq)
    X = np.array(X)
    y = np.array(y)
    # Reshape X to match the expected input shape of the model
    X = np.reshape(X, (X.shape[0], X.shape[1], -1))
    return X, y


def apply_smote(X_train, y_train):
    # Calculate max_seq_len as the length of the longest sequence
    print("Calculating max_seq_len...")
    max_seq_len = max([len(seq) for seq in X_train])

    # Convert list of arrays to 3D numpy array
    X_train = np.stack(
        [
            np.pad(m, ((0, max_seq_len - m.shape[0]), (0, 0)), mode="constant")
            for m in X_train
        ]
    )

    # Reshape to 2D
    print("Reshaping to 2D...")
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape((n_samples * n_timesteps, n_features))

    # Apply SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_reshaped, y_train)

    # Reshape back to 3D
    print("Reshaping back to 3D...")
    X_train_smote = X_train_smote.reshape((-1, n_timesteps, n_features))
    y_train_smote = y_train_smote.reshape((-1, n_timesteps))

    return X_train_smote, y_train_smote


def build_model(max_sequence_length, num_features, X_train, y_train):
    print("Building LSTM model...")

    # set the initial bias to be equal to the logit of the class frequency
    # this helps the model learn faster initially

    initial_bias = np.log([y_train.mean() / (1 - y_train.mean())])
    output_bias = tf.keras.initializers.Constant(initial_bias)

    model = Sequential(name="credit_default_model")

    print("Max sequence length when building the model", max_sequence_length)
    print("Number of features when building the model", num_features)
    model.add(
        LSTM(64, input_shape=(max_sequence_length, num_features), return_sequences=True)
    )
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid", bias_initializer=output_bias))

    optimizer = Adam(learning_rate=0.001)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    model.summary()

    return model


def train_model(model, X_train, y_train):
    print("Training LSTM model...")

    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=1,
        callbacks=[es],
    )
    return model


def plot_loss(history, label, n):
    # Use a log scale on y-axis to show the wide range of values.
    plt.semilogy(
        history.epoch, history.history["loss"], color=colors[n], label="Train " + label
    )
    #   plt.semilogy(history.epoch, history.history['val_loss'],
    #                color=colors[n], label='Val ' + label,
    #                linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()


def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    print("Test Accuracy: ", accuracy_score(y_test, y_pred))
    print("Test Loss: ", log_loss(y_test, y_pred))


def main():
    try:
        df = load_data(
            "/Users/arnav9/Documents/MLE Projects/MLE-capstone/kgl-amx/data/train_data.ftr"
        )
        df = df.iloc[:100000, :]
        df = identify_and_process_categorical_features(df)

        # Replace NaNs with the median value of each column
        df = df.fillna(df.median(numeric_only=True))

        df = df.replace([np.inf, -np.inf], np.finfo(np.float32).max)
        grouped, feature_cols, max_sequence_length = prepare_data(df)

        X, y = prepare_sequences(grouped, feature_cols, max_sequence_length)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        del X, y, df, grouped
        gc.collect()

        # X_train_smote, y_train_smote = apply_smote(X_train, y_train)
        # print("X_train_smote shape", X_train_smote.shape)
        num_features = len(feature_cols)

        model = build_model(max_sequence_length, num_features, X_train, y_train)
        bias_history = train_model(model, X_train, y_train)

        #     es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)
        #     bias_history = model.fit(
        #     X_train,
        #     y_train,
        #     validation_split=0.2,
        #     epochs=100,
        #     batch_size=32,
        #     verbose=1,
        #     callbacks=[es],
        # )
        bias_hx = bias_history.history
        print(bias_hx)
        # plot_loss(bias_history, "Zero Bias", 0)

        # evaluate_model(model, X_test, y_test)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
