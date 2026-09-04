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
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Dropout,
    Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)
import matplotlib.pyplot as plt
import seaborn as sns
import shap
# =====================================================
# PATH
# =====================================================
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "dataset" / "kc_house_data.csv"
RESULT_DIR = BASE_DIR / "result" / "DNN"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_FILE)
# =====================================================
# FEATURES / TARGET
# =====================================================
X = df.iloc[:, 3:]
y = df.iloc[:, 2]
# =====================================================
# TRAIN TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# =====================================================
# SCALE DATA
# =====================================================
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# Save scalers
joblib.dump(x_scaler, RESULT_DIR / "x_scaler.pkl")
joblib.dump(y_scaler, RESULT_DIR / "y_scaler.pkl")

# =====================================================
# BUILD MODEL
# =====================================================
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.30),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.30),

    Dense(64, activation='relu'),
    BatchNormalization(),

    Dense(32, activation='relu'),
    Dense(1)
])

# =====================================================
# COMPILE & TRAIN
# =====================================================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=Huber(),
    metrics=['mae']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
]

history = model.fit(
    X_train_scaled,
    y_train_scaled,
    validation_split=0.10,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Save model in native Keras format
model.save(RESULT_DIR / "DNN_Model.keras")

# =====================================================
# EVALUATION
# =====================================================
y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n====================")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R2   : {r2:.4f}")
print("====================")

# =====================================================
# PERMUTATION IMPORTANCE
# =====================================================
print("\n[INFO] Calculating Permutation Importance...")

class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, keras_model=None, y_scaler=None):
        self.keras_model = keras_model
        self.y_scaler = y_scaler

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        pred_scaled = self.keras_model.predict(X, verbose=0)
        return self.y_scaler.inverse_transform(pred_scaled).flatten()

wrapper_estimator = KerasRegressorWrapper(keras_model=model, y_scaler=y_scaler)

perm_importance = permutation_importance(
    estimator=wrapper_estimator,
    X=X_test_scaled,
    y=y_test,
    scoring='r2',
    n_repeats=5,
    random_state=42,
    n_jobs=1
)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance_Drop_R2": perm_importance.importances_mean,
    "Std_Dev": perm_importance.importances_std
}).sort_values(by="Importance_Drop_R2", ascending=False).reset_index(drop=True)

importance_df.to_excel(RESULT_DIR / "DNN_Feature_Importance.xlsx", index=False)

# Plot Permutation Importance
plt.figure(figsize=(11, 8.5))
sns.set_theme(style="whitegrid")

ax = sns.barplot(
    x="Importance_Drop_R2",
    y="Feature",
    data=importance_df,
    hue="Feature",
    legend=False,
    palette="viridis",
    edgecolor="black",
    linewidth=0.7
)

ax.errorbar(
    x=importance_df["Importance_Drop_R2"],
    y=importance_df["Feature"],
    xerr=importance_df["Std_Dev"],
    fmt='none',
    c='black',
    capsize=3,
    label='Standard Deviation'
)

for i, val in enumerate(importance_df["Importance_Drop_R2"]):
    if val > 0.001:
        ax.text(val + 0.005, i, f"{val:.4f}", va='center', ha='left', fontsize=10, weight='bold')

plt.title("FEATURE IMPORTANCE RANKING (Permutation Importance)", fontsize=13, weight='bold', pad=20)
plt.xlabel("Drop in R² Score upon Permutation", fontsize=12, labelpad=10)
plt.ylabel("House Features", fontsize=12)
plt.legend(loc="lower right")
plt.subplots_adjust(top=0.88, bottom=0.12, left=0.15, right=0.95)
plt.savefig(RESULT_DIR / "DNN_FeatureImportance_Visual.png", dpi=300)
plt.close()

# =====================================================
# SHAP ANALYSIS
# =====================================================
print("\n[INFO] Running SHAP Analysis...")

# 1. Background distribution setup
background_idx = np.random.choice(X_train_scaled.shape[0], 100, replace=False)
background = X_train_scaled[background_idx]
explainer = shap.DeepExplainer(model, background)

# 2. Compute SHAP values
shap_vals = explainer.shap_values(X_test_scaled)

if isinstance(shap_vals, list):
    shap_vals = shap_vals[0]

shap_vals = np.squeeze(shap_vals)

# 3. Rescale SHAP values to actual USD currency impact
y_std = y_scaler.scale_[0]
shap_vals_rescaled = shap_vals * y_std

# 4. Calculate Mean Absolute SHAP per feature in USD
mean_abs_shap = np.abs(shap_vals_rescaled).mean(axis=0)

shap_df = pd.DataFrame({
    "Feature": X.columns,
    "Mean_Absolute_SHAP_USD": mean_abs_shap
}).sort_values(by="Mean_Absolute_SHAP_USD", ascending=False).reset_index(drop=True)

print("\n===== MEAN ABSOLUTE SHAP VALUES (USD) =====")
print(shap_df)
print("===========================================")

shap_df.to_excel(RESULT_DIR / "DNN_SHAP_Importance.xlsx", index=False)

# 5. Plot SHAP visuals
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_vals_rescaled,
    X_test,
    feature_names=X.columns,
    show=False
)
plt.title("SHAP Beeswarm Plot (USD Impact)", fontsize=12, weight='bold', pad=15)
plt.tight_layout()
plt.savefig(RESULT_DIR / "DNN_SHAP_Beeswarm.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_vals_rescaled,
    X_test,
    feature_names=X.columns,
    plot_type="bar",
    show=False
)
plt.title("Feature Importance Ranking (Mean |SHAP Value| in USD)", fontsize=12, weight='bold', pad=15)
plt.tight_layout()
plt.savefig(RESULT_DIR / "DNN_SHAP_Bar.png", dpi=300)
plt.close()

print(f"[INFO] All outputs successfully generated and saved to: {RESULT_DIR}")