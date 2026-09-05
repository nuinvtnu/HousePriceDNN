# =====================================================
# ABLATION STUDY FOR HOUSE PRICE PREDICTION DNN
# =====================================================
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber, MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# PATH & CONFIG
# =====================================================
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "dataset" / "kc_house_data.csv"
RESULT_DIR = BASE_DIR / "result" / "Ablation_Study"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Set random seed để đảm bảo tính công bằng khi so sánh
np.random.seed(42)
import tensorflow as tf

tf.random.set_seed(42)

# =====================================================
# LOAD & PREPROCESS DATA
# =====================================================
df = pd.read_csv(DATA_FILE)
X = df.iloc[:, 3:]
y = df.iloc[:, 2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

input_shape = X_train_scaled.shape[1]


# =====================================================
# BUILD VARIANTS FOR ABLATION STUDY
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
        loss = MeanSquaredError()  # Thay Huber bằng MSE

    elif variant_name == "Shallow_Architecture":
        # Mạng nông: loại bỏ bớt các lớp ẩn (chỉ giữ 128 -> 32)
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
# RUN ABLATION EXPERIMENTAL LOOP
# =====================================================
variants = [
    "Full_Model",
    "No_BatchNorm",
    "No_Dropout",
    "No_Huber_MSE",
    #"Shallow_Architecture"
]

results = []
histories = {}

print("=====================================================")
print("STARTING ABLATION STUDY")
print("=====================================================\n")

for name in variants:
    print(f"--- Training Variant: {name} ---")

    tf.random.set_seed(42)  # Đảm bảo khởi tạo trọng số như nhau nếu có thể
    model = build_model(name)

    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_split=0.10,
        epochs=500,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    # Dự đoán
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

    # Tính toán Metric
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Variant": name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })
    histories[name] = history.history['val_loss']

    print(f"R² Score: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}\n")

# =====================================================
# SAVE RESULTS & EXCEL REPORT
# =====================================================
ablation_df = pd.DataFrame(results)
print("=====================================================")
print("ABLATION STUDY SUMMARY")
print("=====================================================")
print(ablation_df.to_string(index=False))

excel_file = RESULT_DIR / "Ablation_Study_Results.xlsx"
ablation_df.to_excel(excel_file, index=False, sheet_name="Ablation_Metrics")

# =====================================================
# VISUALIZE ABLATION RESULTS
# =====================================================
# 1. R2 Score Comparison Barplot
plt.figure(figsize=(10, 6))
sns.barplot(data=ablation_df, x="R2", y="Variant", palette="viridis")
plt.title("Ablation Study: R² Score Comparison")
plt.xlim(ablation_df["R2"].min() - 0.02, 1.0)
plt.xlabel("R² Score (Higher is Better)")
plt.ylabel("Model Variant")
plt.tight_layout()
plt.savefig(RESULT_DIR / "Ablation_R2_Comparison.png", dpi=300)
plt.close()

# 2. Validation Loss Curves Over Epochs
plt.figure(figsize=(10, 6))
for name, val_loss in histories.items():
    plt.plot(val_loss, label=name)
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Progression across Variants")
plt.legend()
plt.tight_layout()
plt.savefig(RESULT_DIR / "Ablation_ValLoss_Curves.png", dpi=300)
plt.close()

print(f"\nAblation study finished! Results saved in: {RESULT_DIR}")