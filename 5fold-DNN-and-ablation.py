# =====================================================
# ABLATION STUDY WITH 5-FOLD CROSS-VALIDATION
# =====================================================
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber, MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =====================================================
# PATH & CONFIG
# =====================================================
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "dataset" / "kc_house_data.csv"
RESULT_DIR = BASE_DIR / "result" / "Ablation_5Fold_CV"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Set seed
np.random.seed(42)
tf.random.set_seed(42)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_FILE)
X = df.iloc[:, 3:].values
y = df.iloc[:, 2].values.reshape(-1, 1)

input_shape = X.shape[1]


# =====================================================
# BUILD VARIANTS
# =====================================================
def build_model(variant_name):
    model = Sequential()

    if variant_name == "Full_Model":
        model.add(Dense(256, activation='relu', input_shape=(input_shape,)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        loss = Huber()

    elif variant_name == "No_BatchNorm":
        model.add(Dense(256, activation='relu', input_shape=(input_shape,)))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        loss = Huber()

    elif variant_name == "No_Dropout":
        model.add(Dense(256, activation='relu', input_shape=(input_shape,)))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        loss = Huber()

    elif variant_name == "No_Huber_MSE":
        model.add(Dense(256, activation='relu', input_shape=(input_shape,)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        loss = MeanSquaredError()

    elif variant_name == "Shallow_Architecture":
        model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        loss = Huber()

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss,
        metrics=['mae']
    )
    return model


# =====================================================
# 5-FOLD CROSS-VALIDATION LOOP
# =====================================================
variants = [
    "Full_Model",
    "No_BatchNorm",
    "No_Dropout",
    "No_Huber_MSE",
    "Shallow_Architecture"
]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_summary = []

print("=====================================================")
print("STARTING ABLATION STUDY WITH 5-FOLD CROSS-VALIDATION")
print("=====================================================\n")

for variant in variants:
    print(f"\n---> Evaluating Variant: {variant} <---")

    fold_mses = []
    fold_rmses = []
    fold_maes = []
    fold_r2s = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
        X_train_f, X_val_f = X[train_idx], X[val_idx]
        y_train_f, y_val_f = y[train_idx], y[val_idx]

        # Scaling độc lập theo từng Fold (tránh Data Leakage)
        x_scaler = StandardScaler()
        X_train_scaled = x_scaler.fit_transform(X_train_f)
        X_val_scaled = x_scaler.transform(X_val_f)

        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train_f)
        y_val_scaled = y_scaler.transform(y_val_f)

        tf.random.set_seed(42 + fold)
        model = build_model(variant)

        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6)

        model.fit(
            X_train_scaled,
            y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=300,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        # Dự đoán & Inverse Transform
        y_pred_scaled = model.predict(X_val_scaled, verbose=0)
        y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
        y_true = y_val_f.flatten()

        # Calculate fold metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        fold_mses.append(mse)
        fold_rmses.append(rmse)
        fold_maes.append(mae)
        fold_r2s.append(r2)

        print(f"Fold {fold}: R² = {r2:.4f} | MAE = {mae:.2f}")

    # Báo cáo Trung bình ± Độ lệch chuẩn cho mỗi Variant
    cv_summary.append({
        "Variant": variant,
        "MSE (Mean ± Std)": f"{np.mean(fold_mses):.2e} ± {np.std(fold_mses):.2e}",
        "RMSE ($)": f"{np.mean(fold_rmses):.2f} ± {np.std(fold_rmses):.2f}",
        "MAE ($)": f"{np.mean(fold_maes):.2f} ± {np.std(fold_maes):.2f}",
        "R2 Score": f"{np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}",
        "Mean_R2": np.mean(fold_r2s),
        "Std_R2": np.std(fold_r2s)
    })

# =====================================================
# SAVE SUMMARY TO EXCEL
# =====================================================
summary_df = pd.DataFrame(cv_summary)
print("\n=====================================================")
print("5-FOLD CROSS-VALIDATION ABLATION SUMMARY")
print("=====================================================")
print(summary_df[["Variant", "RMSE ($)", "MAE ($)", "R2 Score"]].to_string(index=False))

excel_file = RESULT_DIR / "Ablation_5Fold_CV_Results.xlsx"
summary_df.to_excel(excel_file, index=False, sheet_name="5Fold_CV_Ablation")

print(f"\nCompleted! Saved results to: {excel_file}")