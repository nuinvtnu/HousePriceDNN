# =====================================================
# INDEPENDENT TRAIN & TEST PIPELINE (80/20 SPLIT)
# =====================================================
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Configuration Constants
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset" / "kc_house_data.csv"
RESULT_DIR = BASE_DIR / "result" / "DNN_80_20"

EPOCHS = 500
BATCH_SIZE = 32
TEST_SIZE = 0.20
VAL_SIZE = 0.10
RANDOM_STATE = 42


def load_data(file_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset from CSV file and separate Features (X) and Target (y)."""
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    df = pd.read_csv(file_path)
    X = df.iloc[:, 3:]
    y = df.iloc[:, 2]
    return X, y


def build_model(input_shape: int) -> Sequential:
    """Initialize and compile the Deep Neural Network architecture."""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
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


def get_callbacks() -> List:
    """Return training callbacks for optimization and early stopping."""
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
    return [early_stop, reduce_lr]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression performance evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def save_plots(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        history: dict,
        output_dir: Path
) -> None:
    """Generate and save evaluation plots for the trained model."""
    residuals = y_true - y_pred

    # 1. Scatter Plot (Actual vs Predicted)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Price (Test Set)")
    plt.tight_layout()
    plt.savefig(output_dir / "Test_ScatterPlot.png", dpi=300)
    plt.close()

    # 2. Residual Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.title("Residual Plot (Test Set)")
    plt.tight_layout()
    plt.savefig(output_dir / "Test_ResidualPlot.png", dpi=300)
    plt.close()

    # 3. Error Distribution Plot
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.title("Error Distribution (Test Set)")
    plt.tight_layout()
    plt.savefig(output_dir / "Test_ErrorDistribution.png", dpi=300)
    plt.close()

    # 4. Training History Curve
    if history:
        plt.figure(figsize=(8, 6))
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History Loss')
        plt.tight_layout()
        plt.savefig(output_dir / "TrainingLoss.png", dpi=300)
        plt.close()


def main():
    """Main execution pipeline."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("================ 1. LOADING DATA ================")
    X, y = load_data(DATA_FILE)

    print("================ 2. TRAIN / TEST SPLIT (80/20) ================")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("================ 3. FEATURE SCALING ================")
    # Fit scalers ONLY on training data to prevent Data Leakage
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))

    # Save Scalers
    joblib.dump(x_scaler, RESULT_DIR / "x_scaler.pkl")
    joblib.dump(y_scaler, RESULT_DIR / "y_scaler.pkl")

    print("================ 4. BUILDING & TRAINING MODEL ================")
    model = build_model(input_shape=X_train_scaled.shape[1])

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_split=VAL_SIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(),
        verbose=1
    )

    # Save Trained Model
    model.save(RESULT_DIR / "DNN_Model.h5")

    print("\n================ 5. EVALUATING ON TEST SET ================")
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

    metrics = calculate_metrics(y_test.values, y_pred)

    print("\n---------------- TEST METRICS ----------------")
    for metric_name, value in metrics.items():
        print(f"{metric_name:<6}: {value:.4f}")
    print("----------------------------------------------")

    print("================ 6. EXPORTING RESULTS ================")
    # Export predictions & metrics to Excel
    test_results_df = X_test.copy()
    test_results_df["Actual_Price"] = y_test.values
    test_results_df["Predicted_Price"] = y_pred
    test_results_df["Error"] = y_test.values - y_pred

    metrics_df = pd.DataFrame([metrics])

    excel_file = RESULT_DIR / "DNN_independent.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        test_results_df.to_excel(writer, sheet_name="Predictions", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

    # Save Evaluation Plots
    save_plots(
        y_true=y_test.values,
        y_pred=y_pred,
        history=history.history,
        output_dir=RESULT_DIR
    )

    print(f"\nExecution finished successfully. Results saved in: {RESULT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred during execution: {e}")