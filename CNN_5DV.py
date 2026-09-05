# =====================================================
# HOUSE PRICE PREDICTION USING CNN (1D CONVOLUTIONAL - 5-FOLD CV)
# =====================================================

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
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
# =====================================================
# PATH
# =====================================================
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "dataset" / "kc_house_data.csv"
RESULT_DIR = BASE_DIR / "result" / "CNN_5Fold"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

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
# BUILD CNN MODEL FUNCTION
# =====================================================
def build_cnn_model(input_shape):
    """Hàm khởi tạo cấu trúc mạng CNN 1D mới cho mỗi fold"""
    model = Sequential()

    # Lớp Convolutional 1D nhận vào tensor 3D: (features, 1)
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))

    # Phẳng hóa dữ liệu để chuyển sang các lớp Dense
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output Layer cho bài toán Hồi quy (Regression)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model


# =====================================================
# 5-FOLD CROSS VALIDATION
# =====================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, Y)):
    print(f"\n======================================")
    print(f"RUNNING FOLD {fold + 1} / 5")
    print("======================================")

    # Chia dữ liệu theo Fold
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = Y.iloc[train_idx], Y.iloc[test_idx]

    # --- STANDARDIZATION TRONG TỪNG FOLD ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_test_scaled = scaler.transform(X_test_fold)

    # --- RESHAPE DATA CHO CNN ---
    # Chuyển từ (N, features) thành (N, features, 1)
    X_train_cnn = np.expand_dims(X_train_scaled, axis=-1)
    X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)

    # --- BUILD & TRAIN MODEL ---
    model = build_cnn_model(input_shape=(X_train_cnn.shape[1], 1))

    if fold == 0:
        model.summary()  # In cấu trúc mạng CNN ở fold đầu tiên để kiểm tra

    history = model.fit(
        X_train_cnn,
        y_train_fold,
        validation_split=0.10,
        epochs=100,
        batch_size=32,
        verbose=1
    )

    # --- SAVE MODEL PER FOLD ---
    model_file = RESULT_DIR / f"cnn_house_price_model_Fold{fold + 1}.h5"
    model.save(model_file)
    print(f"✅ CNN Model saved: {model_file}")

    # --- PREDICTION ---
    y_pred = model.predict(X_test_cnn, verbose=0).flatten()

    # --- METRICS PER FOLD ---
    mse = mean_squared_error(y_test_fold, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_fold, y_pred)
    r2 = r2_score(y_test_fold, y_pred)

    print(f"\n===== CNN METRICS FOLD {fold + 1} =====")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    fold_results.append({
        "Fold": f"Fold_{fold + 1}",
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

    # --- SAVE PREDICTION RESULTS & METRICS TO EXCEL ---
    results_df = X_test_fold.copy()
    results_df["y_test"] = y_test_fold.values
    results_df["y_pred"] = y_pred

    metrics_df = pd.DataFrame({
        "Metric": ["MSE", "RMSE", "MAE", "R2"],
        "Value": [mse, rmse, mae, r2]
    })

    excel_file = RESULT_DIR / f"CNN_Result_Fold{fold + 1}.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Prediction", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
    print(f"✅ Excel saved: {excel_file}")

    # --- SCATTER PLOT ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test_fold, y=y_pred, alpha=0.6)
    plt.plot([y_test_fold.min(), y_test_fold.max()], [y_test_fold.min(), y_test_fold.max()], 'r--', linewidth=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"CNN Regression Fold {fold + 1} (R² = {r2:.4f})")
    plt.grid(True)
    plt.tight_layout()

    scatter_file = RESULT_DIR / f"CNN_ScatterPlot_Fold{fold + 1}.png"
    plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
    plt.close()

    # --- RESIDUAL PLOT ---
    residuals = y_test_fold - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot Fold {fold + 1}")
    plt.grid(True)
    plt.tight_layout()

    residual_file = RESULT_DIR / f"CNN_ResidualPlot_Fold{fold + 1}.png"
    plt.savefig(residual_file, dpi=300, bbox_inches='tight')
    plt.close()

    # --- ERROR DISTRIBUTION ---
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Residual")
    plt.title(f"Residual Distribution Fold {fold + 1}")
    plt.tight_layout()

    error_file = RESULT_DIR / f"CNN_ErrorDistribution_Fold{fold + 1}.png"
    plt.savefig(error_file, dpi=300, bbox_inches='tight')
    plt.close()

    # --- TRAINING LOSS CURVE ---
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training History Fold {fold + 1}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_file = RESULT_DIR / f"CNN_TrainingLoss_Fold{fold + 1}.png"
    plt.savefig(loss_file, dpi=300, bbox_inches='tight')
    plt.close()

# =====================================================
# SUMMARY TABLE & EXPORT
# =====================================================
summary_df = pd.DataFrame(fold_results)

# Tính trung bình (Mean) và độ lệch chuẩn (Std) của cả 5 fold
mean_row = summary_df.mean(numeric_only=True).to_dict()
mean_row["Fold"] = "Mean"

std_row = summary_df.std(numeric_only=True).to_dict()
std_row["Fold"] = "Std"

summary_df = pd.concat([summary_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

# Đẩy cột "Fold" lên trước
cols = ["Fold"] + [col for col in summary_df.columns if col != "Fold"]
summary_df = summary_df[cols]

summary_file = RESULT_DIR / "Summary_CNN_5Fold_Final.xlsx"
summary_df.to_excel(summary_file, index=False)

print("\n====================================")
print("ALL FOLDS FINISHED SUCCESSFULLY WITH CNN!")
print("====================================")
print(summary_df)
print(f"\nSummary results saved to: {summary_file}")
print(f"All images and individual sheets are inside: {RESULT_DIR}")