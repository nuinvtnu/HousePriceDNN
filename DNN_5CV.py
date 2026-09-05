# =====================================================
# HOUSE PRICE PREDICTION USING IMPROVED DNN (5-FOLD CV)
# =====================================================
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Configuration Constants
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset" / "kc_house_data.csv"
RESULT_DIR = BASE_DIR / "result" / "DNN_5CV"
N_SPLITS = 5
EPOCHS = 500
BATCH_SIZE = 32
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
    """Return a list of training callbacks for model optimization."""
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
    """Calculate performance evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def save_plots(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        history: dict,
        output_dir: Path,
        prefix: str = "CV"
) -> None:
    """Generate and save visualization plots for evaluation results."""
    residuals = y_true - y_pred

    # 1. Scatter Plot (Actual vs Predicted)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs Predicted ({prefix})")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_ScatterPlot.png", dpi=300)
    plt.close()

    # 2. Residual Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot ({prefix})")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_ResidualPlot.png", dpi=300)
    plt.close()

    # 3. Error Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.title(f"Error Distribution ({prefix})")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_ErrorDistribution.png", dpi=300)
    plt.close()

    # 4. Training History Curve (for the best model fold)
    if history:
        plt.figure(figsize=(8, 6))
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Training History ({prefix})')
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_TrainingLoss.png", dpi=300)
        plt.close()


def train_and_evaluate_cv(
        X: pd.DataFrame,
        y: pd.Series,
        output_dir: Path,
        n_splits: int = 5
) -> None:
    """Perform 5-Fold Cross Validation pipeline for the DNN model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics: List[Dict[str, float]] = []
    oof_predictions = np.zeros(len(X))
    best_r2 = -float('inf')

    print(f"\n================ START {n_splits}-FOLD CROSS VALIDATION ================")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
        print(f"\n--- Training Fold {fold}/{n_splits} ---")

        # Split Train/Validation data for the current fold
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit Scaler ON TRAIN SET ONLY AND TRANSFORM (Prevents Data Leakage)
        x_scaler = StandardScaler()
        X_train_scaled = x_scaler.fit_transform(X_train)
        X_val_scaled = x_scaler.transform(X_val)

        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))

        # Build & Train Model
        model = build_model(input_shape=X_train_scaled.shape[1])
        history = model.fit(
            X_train_scaled,
            y_train_scaled,
            validation_data=(X_val_scaled, y_scaler.transform(y_val.values.reshape(-1, 1))),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=get_callbacks(),
            verbose=0
        )

        # Predict on Validation set
        y_pred_scaled = model.predict(X_val_scaled, verbose=0)
        y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

        # Save Out-Of-Fold (OOF) predictions
        oof_predictions[val_idx] = y_pred

        # Calculate metrics for the current fold
        metrics = calculate_metrics(y_val.values, y_pred)
        fold_metrics.append(metrics)

        print(f"Fold {fold} - MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | R2: {metrics['R2']:.4f}")

        # Save Scalers and Model belonging to the fold with the highest R2 score
        if metrics['R2'] > best_r2:
            best_r2 = metrics['R2']
            model.save(output_dir / "best_DNN_model.h5")
            joblib.dump(x_scaler, output_dir / "best_x_scaler.pkl")
            joblib.dump(y_scaler, output_dir / "best_y_scaler.pkl")
            best_history = history.history

    # =====================================================
    # Aggregate & Export Cross-Validation Summary Report
    # =====================================================
    cv_metrics_df = pd.DataFrame(fold_metrics)
    summary_metrics = pd.DataFrame({
        "Mean": cv_metrics_df.mean(),
        "Std": cv_metrics_df.std()
    }).T

    print("\n================ CROSS VALIDATION SUMMARY ================")
    print(cv_metrics_df)
    print("\nMean & Std across Folds:")
    print(summary_metrics)
    print("==========================================================")

    # Export results to Excel
    oof_df = X.copy()
    oof_df["y_actual"] = y.values
    oof_df["y_pred_oof"] = oof_predictions

    excel_file = output_dir / "DNN_CV_Results.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        oof_df.to_excel(writer, sheet_name="OOF_Predictions", index=False)
        cv_metrics_df.to_excel(writer, sheet_name="Fold_Metrics", index_label="Fold")
        summary_metrics.to_excel(writer, sheet_name="Summary")

    # Generate overall plots based on complete Out-Of-Fold predictions
    save_plots(
        y_true=y.values,
        y_pred=oof_predictions,
        history=best_history,
        output_dir=output_dir,
        prefix="Overall_CV"
    )

    print(f"\nAll execution results have been saved to: {output_dir}")


def main():
    """Main execution function of the script."""
    try:
        X, y = load_data(DATA_FILE)
        train_and_evaluate_cv(
            X=X,
            y=y,
            output_dir=RESULT_DIR,
            n_splits=N_SPLITS
        )
    except Exception as e:
        print(f"An error occurred during execution: {e}")


if __name__ == "__main__":
    main()