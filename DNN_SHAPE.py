# =====================================================
# HOUSE PRICE PREDICTION USING IMPROVED DNN (WITH FEATURE IMPORTANCE)
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
from sklearn.inspection import permutation_importance  # <-- THÊM MỚI

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
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

# =====================================================
# SCALE Y
# =====================================================
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# =====================================================
# SAVE SCALERS
# =====================================================
joblib.dump(x_scaler, RESULT_DIR / "x_scaler.pkl")
joblib.dump(y_scaler, RESULT_DIR / "y_scaler.pkl")

# =====================================================
# BUILD MODEL
# =====================================================
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
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

# =====================================================
# COMPILE
# =====================================================
model.compile(
    optimizer=Adam(learning_rate=0.001),
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
    epochs=100,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# =====================================================
# SAVE MODEL
# =====================================================
model.save(RESULT_DIR / "DNN_Model.h5")

# =====================================================
# PREDICTION
# =====================================================
y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

# =====================================================
# METRICS
# =====================================================
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
## =====================================================
# NEW: FEATURE IMPORTANCE ANALYSIS (Phân tích yếu tố ảnh hưởng)
# =====================================================
print("\n[INFO] Đang phân tích các yếu tố ảnh hưởng đến giá nhà (Permutation Importance)...")

from sklearn.base import BaseEstimator, RegressorMixin # <-- THÊM THƯ VIỆN NÀY

# Tạo Class kế thừa chuẩn từ Scikit-Learn để tránh lỗi __sklearn_tags__
class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, keras_model=None, y_scaler=None):
        self.keras_model = keras_model
        self.y_scaler = y_scaler

    def fit(self, X, y=None):
        # Mô hình DNN đã train xong trước đó nên không cần fit lại
        return self

    def predict(self, X):
        # Dự đoán từ mạng DNN (đầu ra đang bị scale)
        pred_scaled = self.keras_model.predict(X, verbose=0)
        # Đảo ngược scale về giá trị thực tế của giá nhà
        return self.y_scaler.inverse_transform(pred_scaled).flatten()

# Khởi tạo đối tượng bọc hợp lệ
wrapper_estimator = KerasRegressorWrapper(keras_model=model, y_scaler=y_scaler)

# Tính toán Permutation Importance dựa trên điểm R2 score của tập Test thực tế
perm_importance = permutation_importance(
    estimator=wrapper_estimator,
    X=X_test_scaled,
    y=y_test,
    scoring='r2',
    n_repeats=5,
    random_state=42,
    n_jobs=1
)

# Tạo DataFrame lưu trữ kết quả quan trọng
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance_Drop_R2": perm_importance.importances_mean,
    "Std_Dev": perm_importance.importances_std
})

# Sắp xếp theo thứ tự giảm dần (Yếu tố ảnh hưởng nhiều nhất nằm ở đầu)
importance_df = importance_df.sort_values(by="Importance_Drop_R2", ascending=False).reset_index(drop=True)

print("\n===== ĐỘ QUAN TRỌNG CỦA CÁC YẾU TỐ (XẾP GIẢM DẦN) =====")
print(importance_df)
print("======================================================")

# =====================================================
# NEW: EXPORT TO EXCEL (Bổ sung phần ghi file Excel)
# =====================================================
excel_importance_file = RESULT_DIR / "DNN_Feature_Importance.xlsx"
importance_df.to_excel(excel_importance_file, index=False, sheet_name="Feature_Importance")
print(f"[INFO] Bảng số liệu yếu tố ảnh hưởng đã được lưu tại: {excel_importance_file}")

# =====================================================
# NEW: PLOT FEATURE IMPORTANCE (ENGLISH VERSION)
# =====================================================
print("[INFO] Generating Feature Importance plot...")

# 1. Configure the plot layout and style (Chỉnh lại size và adjust tránh che khuất tiêu đề)
plt.figure(figsize=(11, 8.5))
sns.set_theme(style="whitegrid")

# 2. Create horizontal bar plot directly from the importance_df
ax = sns.barplot(
    x="Importance_Drop_R2",
    y="Feature",
    data=importance_df,
    palette="viridis",
    edgecolor="black",
    linewidth=0.7
)

# 3. Add Error Bars representing Standard Deviation (Std Dev)
ax.errorbar(
    x=importance_df["Importance_Drop_R2"],
    y=importance_df["Feature"],
    xerr=importance_df["Std_Dev"],
    fmt='none',
    c='black',
    capsize=3,
    label='Standard Deviation (Std Dev)'
)

# 4. Automatically annotate the exact values at the end of each bar
for i, val in enumerate(importance_df["Importance_Drop_R2"]):
    if val > 0.001:
        ax.text(
            val + 0.005,
            i,
            f"{val:.4f}",
            va='center',
            ha='left',
            fontsize=10,
            weight='bold'
        )

# 5. Set Titles and Labels in academic English
plt.title(
    "FEATURE IMPORTANCE RANKING FOR HOUSE PRICE PREDICTION\n(Permutation Importance Method on DNN Model)",
    fontsize=13,
    weight='bold',
    pad=20
)
plt.xlabel(
    "Drop in R² Score upon Feature Permutation (Higher is more important)",
    fontsize=12,
    labelpad=10
)
plt.ylabel(
    "House Features",
    fontsize=12
)

# Add Legend for the error bars
plt.legend(loc="lower right")

# SỬA LỖI CHE KHUẤT TIÊU ĐỀ: Định hình lại khoảng cách lề trên
plt.subplots_adjust(top=0.88, bottom=0.12, left=0.15, right=0.95)

# Save the high-res figure
plt.savefig(RESULT_DIR / "DNN_FeatureImportance_Visual.png", dpi=300)
plt.close()

print(f"[INFO] Feature Importance plot saved successfully at: {RESULT_DIR / 'DNN_FeatureImportance_Visual.png'}")