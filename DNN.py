# =====================================================

# HOUSE PRICE PREDICTION USING IMPROVED DNN

# =====================================================

import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
mean_squared_error,
mean_absolute_error,
r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
Dense,
BatchNormalization,
Dropout
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

from tensorflow.keras.callbacks import (
EarlyStopping,
ReduceLROnPlateau
)

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================

# PATH

# =====================================================

BASE_DIR = Path(__file__).parent

DATA_FILE = BASE_DIR / "dataset" / "kc_house_data.csv"

RESULT_DIR = BASE_DIR / "result" / "DNN"

RESULT_DIR.mkdir(
parents=True,
exist_ok=True
)

# =====================================================

# LOAD DATA

# =====================================================

df = pd.read_csv(DATA_FILE)

print(df.head())

# =====================================================

# FEATURES / TARGET

# =====================================================

X = df.iloc[:, 3:]

y = df.iloc[:, 2]

# =====================================================

# TRAIN TEST SPLIT

# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.20,
random_state=42
)

# =====================================================

# SCALE X

# =====================================================

x_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(
X_train
)

X_test_scaled = x_scaler.transform(
X_test
)

# =====================================================

# SCALE Y

# =====================================================

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
y_train.values.reshape(-1, 1)
)

y_test_scaled = y_scaler.transform(
y_test.values.reshape(-1, 1)
)

# =====================================================

# SAVE SCALERS

# =====================================================

joblib.dump(
x_scaler,
RESULT_DIR / "x_scaler.pkl"
)

joblib.dump(
y_scaler,
RESULT_DIR / "y_scaler.pkl"
)

# =====================================================

# BUILD MODEL

# =====================================================

model = Sequential([

Dense(
    256,
    activation='relu',
    input_shape=(
        X_train_scaled.shape[1],
    )
),

BatchNormalization(),
Dropout(0.20),

Dense(
    128,
    activation='relu'
),

BatchNormalization(),
Dropout(0.20),

Dense(
    64,
    activation='relu'
),

BatchNormalization(),

Dense(
    32,
    activation='relu'
),

Dense(1)


])

# =====================================================

# COMPILE

# =====================================================

model.compile(
optimizer=Adam(
learning_rate=0.001
),
loss=Huber(),
metrics=['mae']
)
model.summary()
# =====================================================

# CALLBACKS

# =====================================================

early_stop = EarlyStopping(
monitor='val_loss',
patience=20,
restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
monitor='val_loss',
factor=0.5,
patience=8,
min_lr=1e-6
)

# =====================================================

# TRAIN

# =====================================================

history = model.fit(
X_train_scaled,
y_train_scaled,
validation_split=0.10,
epochs=500,
batch_size=64,
callbacks=[
early_stop,
reduce_lr
],
verbose=1
)

# =====================================================

# SAVE MODEL

# =====================================================

model.save(
RESULT_DIR /
"DNN_Model.h5"
)

# =====================================================

# PREDICTION

# =====================================================

y_pred_scaled = model.predict(
X_test_scaled,
verbose=0
)

y_pred = y_scaler.inverse_transform(
y_pred_scaled
).flatten()

# =====================================================

# METRICS

# =====================================================

mse = mean_squared_error(
y_test,
y_pred
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
y_test,
y_pred
)

r2 = r2_score(
y_test,
y_pred
)

print("\n====================")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R2   : {r2:.4f}")
print("====================")

# =====================================================

# SAVE PREDICTION

# =====================================================

results_df = X_test.copy()

results_df["y_test"] = y_test.values

results_df["y_pred"] = y_pred

metrics_df = pd.DataFrame({
"Metric": [
"MSE",
"RMSE",
"MAE",
"R2"
],
"Value": [
mse,
rmse,
mae,
r2
]
})

# =====================================================

# SAVE EXCEL

# =====================================================

excel_file = (
RESULT_DIR /
"DNN_Result.xlsx"
)

with pd.ExcelWriter(
excel_file,
engine="openpyxl"
) as writer:

    results_df.to_excel(
        writer,
        sheet_name="Prediction",
        index=False
    )

    metrics_df.to_excel(
        writer,
        sheet_name="Metrics",
        index=False
    )


# =====================================================

# SCATTER PLOT

# =====================================================

plt.figure(figsize=(8,6))

sns.scatterplot(
x=y_test,
y=y_pred,
alpha=0.6
)

plt.plot(
[y_test.min(), y_test.max()],
[y_test.min(), y_test.max()],
'r--'
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title(
f"DNN (R²={r2:.4f})"
)

plt.tight_layout()

plt.savefig(
RESULT_DIR /
"DNN_ScatterPlot.png",
dpi=300
)

plt.close()

# =====================================================

# RESIDUAL PLOT

# =====================================================

residuals = y_test - y_pred

plt.figure(figsize=(8,6))

sns.scatterplot(
x=y_pred,
y=residuals,
alpha=0.6
)

plt.axhline(
y=0,
color='red',
linestyle='--'
)

plt.xlabel("Predicted")
plt.ylabel("Residual")

plt.title(
"Residual Plot"
)

plt.tight_layout()

plt.savefig(
RESULT_DIR /
"DNN_ResidualPlot.png",
dpi=300
)

plt.close()

# =====================================================

# ERROR DISTRIBUTION

# =====================================================

plt.figure(figsize=(8,6))

sns.histplot(
residuals,
kde=True
)

plt.title(
"Error Distribution"
)

plt.tight_layout()

plt.savefig(
RESULT_DIR /
"DNN_ErrorDistribution.png",
dpi=300
)

plt.close()

# =====================================================

# TRAINING CURVE

# =====================================================

plt.figure(figsize=(8,6))

plt.plot(
history.history["loss"],
label="Train Loss"
)

plt.plot(
history.history["val_loss"],
label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.title(
"Training History"
)

plt.tight_layout()

plt.savefig(
RESULT_DIR /
"DNN_TrainingLoss.png",
dpi=300
)

plt.close()

print("\n================================")
print("DNN FINISHED")
print("================================")
print(f"Result folder: {RESULT_DIR}")

