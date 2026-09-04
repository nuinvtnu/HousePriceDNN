# =====================================================
# HOUSE PRICE PREDICTION USING DEEP NEURAL NETWORK
# =====================================================
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
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
# =====================================================
# PATH
# =====================================================
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "dataset" / "kc_house_data.csv"
RESULT_DIR = BASE_DIR / "result"
RESULT_DIR.mkdir(exist_ok=True)
# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_FILE)
print("\n===== DATASET =====")
print(df.head())
# feature columns
X = df.iloc[:, 3:]
# target column (price)
Y = df.iloc[:, 2]
print("\n===== FEATURE SAMPLE =====")
print(X.head())
print("\n===== TARGET SAMPLE =====")
print(Y.head())
# =====================================================
# TRAIN TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)
# =====================================================
# STANDARDIZATION
# =====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# =====================================================
# BUILD MODEL
# =====================================================
model = Sequential()
model.add(
    Dense(
        128,
        activation='relu',
        input_shape=(X_train_scaled.shape[1],)
    )
)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
# =====================================================
# COMPILE
# =====================================================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
# =====================================================
# TRAIN
# =====================================================
history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.10,
    epochs=100,
    batch_size=32,
    verbose=1
)
# =====================================================
# SAVE MODEL
# =====================================================
model_file = RESULT_DIR / "house_price_model.h5"
model.save(model_file)
print(f"\n✅ Model saved: {model_file}")
# =====================================================
# EVALUATION
# =====================================================
loss, mae = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)
# =====================================================
# PREDICTION
# =====================================================
y_pred = model.predict(
    X_test_scaled,
    verbose=0
).flatten()
# =====================================================
# METRICS
# =====================================================
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\n===== METRICS =====")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")
# =====================================================
# SAVE PREDICTION RESULTS
# =====================================================
results_df = X_test.copy()
results_df["y_test"] = y_test.values
results_df["y_pred"] = y_pred
# =====================================================
# SAVE METRICS
# =====================================================
metrics_df = pd.DataFrame({
    "Metric": ["MSE", "RMSE", "MAE", "R2"],
    "Value": [mse, rmse, mae, r2]
})
# =====================================================
# EXPORT EXCEL
# =====================================================
excel_file = RESULT_DIR / "CNN_Result.xlsx"
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
print(f"\n✅ Excel saved: {excel_file}")
# =====================================================
# SCATTER PLOT
# =====================================================
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=y_test,
    y=y_pred,
    alpha=0.6
)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2
)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(
    f"CNN Regression (R² = {r2:.4f})"
)
plt.grid(True)
plt.tight_layout()
scatter_file = RESULT_DIR / "CNN_ScatterPlot.png"
plt.savefig(
    scatter_file,
    dpi=300,
    bbox_inches='tight'
)
plt.close()
print(f" Saved: {scatter_file}")
# =====================================================
# RESIDUAL PLOT
# =====================================================
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
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
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.grid(True)
plt.tight_layout()
residual_file = RESULT_DIR / "CNN_ResidualPlot.png"
plt.savefig(
    residual_file,
    dpi=300,
    bbox_inches='tight'
)
plt.close()
print(f" Saved: {residual_file}")
# =====================================================
# ERROR DISTRIBUTION
# =====================================================
plt.figure(figsize=(8, 6))
sns.histplot(
    residuals,
    kde=True
)
plt.xlabel("Residual")
plt.title("Residual Distribution")
plt.tight_layout()
error_file = RESULT_DIR / "CNN_ErrorDistribution.png"
plt.savefig(
    error_file,
    dpi=300,
    bbox_inches='tight'
)
plt.close()
print(f" Saved: {error_file}")
# =====================================================
# TRAINING LOSS CURVE
# =====================================================
plt.figure(figsize=(8, 6))
plt.plot(
    history.history['loss'],
    label='Training Loss'
)
plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_file = RESULT_DIR / "CNN_TrainingLoss.png"
plt.savefig(
    loss_file,
    dpi=300,
    bbox_inches='tight'
)
plt.close()

print(f" Saved: {loss_file}")
print("\n====================================")
print("Finished!")
print(f"Results folder: {RESULT_DIR}")
print("====================================")