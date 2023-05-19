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
df.drop(["date", "Month", "Day", "Year"], axis=1, inplace=True)

# Sorting by date and customer_ID
df.sort_values(["customer_ID", "Date"], inplace=True)

# Dropping customer_ID and Date for model training
model_df = df.drop(columns=["Date"])


print(f"One-hot encoding categorical features")
# categorical_columns = model_df.select_dtypes(include=["object"]).columns
categorical_columns = ["D_63", "D_64"]
model_df = pd.get_dummies(model_df, columns=categorical_columns)

float16_columns = model_df.select_dtypes(include=["float64"]).columns

scaler = StandardScaler()
model_df[float16_columns] = scaler.fit_transform(model_df[float16_columns])

print(f"Normalizing numerical features")
# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
model_df[model_df.columns.difference(["customer_ID"])] = scaler.fit_transform(
    model_df[model_df.columns.difference(["customer_ID"])]
)

print(f"Replacing NaNs and infinite values")
model_df.fillna(
    df.median(), inplace=True
)  # Replace NaNs with the median value of each column
model_df.replace(
    [np.inf, -np.inf], np.finfo(np.float32).max, inplace=True
)  # Replace infinite values with the max float32 value

print("Creating sequences")
# Create sequences for each customer_ID
sequence_data = model_df.groupby("customer_ID").apply(
    lambda x: x[model_df.columns.difference(["customer_ID"])].values
)

# Pad sequences with 0's
padded_sequences = pad_sequences(
    sequence_data, maxlen=13, dtype="float32", padding="post"
)

print("Defining features (X) and target (y)")
# Defining features (X) and target (y)
X = padded_sequences[:, :, :-1]  # all data except the last column (target)
y = padded_sequences[:, -1, -1]  # last month's target

print("Splitting data into train and test sets")
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
train_pred = model.predict(X_train)

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
# Predict probabilities for the test set
test_pred = model.predict(X_test)

# Calculating AUC score
print("Test Set ROC AUC score:", roc_auc_score(y_test, test_pred))

# Plotting ROC Curve for the test set
fpr, tpr, _ = roc_curve(y_test, test_pred)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc_score(y_test, test_pred))
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic Curve")
plt.legend(loc="lower right")
plt.show()

# %%
from sklearn.metrics import precision_score, recall_score, f1_score

# Choose a threshold
threshold = 0.7

# Convert predicted probabilities to class labels
test_pred_labels = (test_pred > threshold).astype(int)

# Calculate Precision
precision = precision_score(y_test, test_pred_labels)
print("Precision: ", precision)

# Calculate Recall
recall = recall_score(y_test, test_pred_labels)
print("Recall: ", recall)

# Calculate F1 Score
f1 = f1_score(y_test, test_pred_labels)
print("F1 Score: ", f1)

# %%
