# =====================================================
# HOUSE PRICE PREDICTION USING IMPROVED DNN (5-FOLD CV)
# =====================================================

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold
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
RESULT_DIR = BASE_DIR / "result" / "DNN_5Fold"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_FILE)
print("Data head:")
print(df.head())

# Feature columns và Target column
X = df.iloc[:, 3:]
y = df.iloc[:, 2]


# =====================================================
# BUILD MODEL FUNCTION
# =====================================================
def build_dnn_model(input_dim):
    """Hàm khởi tạo lại kiến trúc mô hình mới cho mỗi fold"""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.20),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.20),

        Dense(64, activation='relu'),
        BatchNormalization(),

        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=Huber(),
        metrics=['mae']
    )
    return model


# =====================================================
# 5-FOLD CROSS VALIDATION
# =====================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

# Mảng lưu dự đoán Out-of-Fold để đánh giá tổng thể nếu cần
oof_preds = np.zeros(len(X))

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    print(f"\n======================================")
    print(f"RUNNING FOLD {fold + 1} / 5")
    print("======================================")

    # Chia dữ liệu theo Fold (giữ nguyên định dạng Pandas cho tập test để lưu file)
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

    # --- SCALE DATA TRONG TỪNG FOLD ---
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_fold)
    X_test_scaled = x_scaler.transform(X_test_fold)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_fold.values.reshape(-1, 1))

    # Lưu lại bộ scaler của từng fold
    joblib.dump(x_scaler, RESULT_DIR / f"x_scaler_fold{fold + 1}.pkl")
    joblib.dump(y_scaler, RESULT_DIR / f"y_scaler_fold{fold + 1}.pkl")

    # --- BUILD & TRAIN MODEL ---
    model = build_dnn_model(X_train_scaled.shape[1])

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

    # Sử dụng validation_split từ chính tập train của fold
    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_split=0.10,
        epochs=500,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Lưu model của fold hiện tại
    model.save(RESULT_DIR / f"DNN_Model_Fold{fold + 1}.h5")

    # --- PREDICTION ---
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

    # Lưu vào mảng OOF toàn cục
    oof_preds[test_idx] = y_pred

    # --- EVALUATE METRICS FOR FOLD ---
    mse = mean_squared_error(y_test_fold, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_fold, y_pred)
    r2 = r2_score(y_test_fold, y_pred)

    print(f"\nFold {fold + 1} Metrics:")
    print(f"MSE  : {mse:.4f} | RMSE : {rmse:.4f} | MAE  : {mae:.4f} | R2   : {r2:.4f}")

    # Lưu metric của fold vào danh sách chung
    fold_results.append({
        "Fold": f"Fold_{fold + 1}",
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

    # --- SAVE EXCEL FOR FOLD ---
    results_df = X_test_fold.copy()
    results_df["y_test"] = y_test_fold.values
    results_df["y_pred"] = y_pred

    metrics_df = pd.DataFrame({
        "Metric": ["MSE", "RMSE", "MAE", "R2"],
        "Value": [mse, rmse, mae, r2]
    })

    excel_file = RESULT_DIR / f"DNN_Result_Fold{fold + 1}.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Prediction", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

    # --- PLOTS FOR FOLD ---
    # 1. Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test_fold, y=y_pred, alpha=0.6)
    plt.plot([y_test_fold.min(), y_test_fold.max()], [y_test_fold.min(), y_test_fold.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"DNN Fold {fold + 1} (R²={r2:.4f})")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"DNN_ScatterPlot_Fold{fold + 1}.png", dpi=300)
    plt.close()

    # 2. Residual Plot
    residuals = y_test_fold - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot Fold {fold + 1}")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"DNN_ResidualPlot_Fold{fold + 1}.png", dpi=300)
    plt.close()

    # 3. Training Curve
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Training History Fold {fold + 1}")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"DNN_TrainingLoss_Fold{fold + 1}.png", dpi=300)
    plt.close()

# =====================================================
# SUMMARY TABLE & AVERAGE
# =====================================================
summary_df = pd.DataFrame(fold_results)

# Tính dòng trung bình (Mean) và độ lệch chuẩn (Std) của cả 5 fold
mean_row = summary_df.mean(numeric_only=True).to_dict()
mean_row["Fold"] = "Mean"

std_row = summary_df.std(numeric_only=True).to_dict()
std_row["Fold"] = "Std"

# Gộp vào bảng tổng kết
summary_df = pd.concat([summary_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

# Lưu bảng tổng kết ra Excel
summary_file = RESULT_DIR / "Summary_DNN_5Fold.xlsx"
summary_df.to_excel(summary_file, index=False)

print("\n======================================")
print("5-FOLD CROSS VALIDATION FINISHED")
print("======================================")
print(summary_df)
print(f"\nAll results saved to folder: {RESULT_DIR}")