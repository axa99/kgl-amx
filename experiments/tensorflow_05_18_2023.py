# %%


import pandas as pd
import numpy as np
from tqdm import tqdm


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import pad_sequences
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


# Define constants
MAX_SEQUENCE_LENGTH = 13

print("Loading data...")
df = pd.read_feather("../data/train_data.ftr")

# Preserve 'customer_ID' for grouping
print("Preserving 'customer_ID' for grouping")
customer_ids = df["customer_ID"]

print("Processing 'S_2' column...")
# Convert 'S_2' to datetime
df["S_2"] = pd.to_datetime(df["S_2"])

# Create a baseline date
baseline_date = df["S_2"].min()

# Convert 'S_2' to number of days since baseline date
df["S_2"] = (df["S_2"] - baseline_date).dt.days


# Determine categorical features
print("Determining categorical features...")
cat_cols = df.select_dtypes(include=["category"]).columns

# Preprocess categorical features
print("Preprocessing categorical features...")
for col in tqdm(cat_cols):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Add 'customer_ID' back to the dataframe
df["customer_ID"] = customer_ids

print("Grouping by customer_ID...")
grouped = df.groupby(["customer_ID"])

# Prepare sequences and pad them
print("Preparing and padding sequences...")
X = pad_sequences(
    grouped.apply(lambda x: x.values.tolist()),
    maxlen=MAX_SEQUENCE_LENGTH,
    dtype="float32",
)

# Separate features and target variable
print("Separating features and target variable...")
y = df["target"].values

# Split data into training and test sets
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize the data
print("Standardizing the data...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Use SMOTE to balance the data
print("Balancing the data with SMOTE...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Define model architecture
print("Defining model architecture...")


def create_model(optimizer="adam"):
    model = Sequential()
    model.add(
        LSTM(
            100,
            activation="relu",
            return_sequences=True,
            input_shape=(X_train.shape[1], 1),
        )
    )
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model, verbose=0)

# Grid search parameters
batch_size = [10, 20, 50, 100]
epochs = [10, 50, 100]
optimizer = ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"]
param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

# Fit the model
print("Fitting the model...")
grid_result = grid.fit(
    X_train,
    y_train,
    callbacks=[EarlyStopping(monitor="val_loss", patience=3, min_delta=0.0001)],
)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Evaluate model
print("Evaluating model...")
score = grid.score(X_test, y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# %%
