# %%
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# %%
# Assuming that data is loaded in df dataframe
df = pd.read_csv("../data/train_rndm_sample.csv")

df.rename(columns={"S_2": "date"}, inplace=True)

# Converting date to datetime format
df["Date"] = pd.to_datetime(df["date"])
df.drop(["date"], axis=1, inplace=True)

# Sorting by date and customer_ID
df.sort_values(["customer_ID", "Date"], inplace=True)

# Dropping customer_ID and Date for model training
model_df = df.drop(columns=["Date"])


# One-hot encoding categorical features
categorical_columns = model_df.select_dtypes(include=["category"]).columns
categorical_columns
model_df = pd.get_dummies(model_df, columns=categorical_columns)

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
model_df[model_df.columns.difference(["customer_ID"])] = scaler.fit_transform(
    model_df[model_df.columns.difference(["customer_ID"])]
)

# Create sequences for each customer_ID
sequence_data = model_df.groupby("customer_ID").apply(
    lambda x: x[model_df.columns.difference(["customer_ID"])].values
)

# Pad sequences with 0's
padded_sequences = pad_sequences(
    sequence_data, maxlen=13, dtype="float32", padding="post"
)

# Defining features (X) and target (y)
X = padded_sequences[:, :, :-1]  # all data except the last column (target)
y = padded_sequences[:, :, -1:]  # only the last column (target)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Defining the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation="sigmoid"))

# Compiling the model
model.compile(loss="binary_crossentropy", optimizer="adam")

# Fitting the model
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=72,
    validation_data=(X_test, y_test),
    verbose=2,
    shuffle=False,
)

# Predict probabilities for the training set
train_pred = model.predict_proba(X_train)

# Calculating AUC score
print("Training Set ROC AUC score:", roc_auc_score(y_train, train_pred))

# Plotting ROC Curve
fpr, tpr, _ = roc_curve(y_train, train_pred)
plt.figure(figsize=(10, 8))
plt.plot(
    fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc_score(y_train, train_pred)
)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic Curve")
plt.legend(loc="lower right")
plt.show()

# %%
