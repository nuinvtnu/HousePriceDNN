import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import matplotlib.pyplot as plt
import seaborn as sns


def save_regression_results(
        model,
        model_name,
        X_test,
        y_test,
        y_pred,
        result_root="result"
):

    # =====================================
    # CREATE FOLDER
    # =====================================
    model_dir = Path(result_root) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # =====================================
    # SAVE MODEL
    # =====================================
    joblib.dump(
        model,
        model_dir / f"{model_name}_Model.pkl"
    )

    # =====================================
    # METRICS
    # =====================================
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

    # =====================================
    # SAVE PREDICTIONS
    # =====================================
    results_df = X_test.copy()

    results_df["y_test"] = y_test.values
    results_df["y_pred"] = y_pred

    metrics_df = pd.DataFrame({
        "Metric": ["MSE", "RMSE", "MAE", "R2"],
        "Value": [mse, rmse, mae, r2]
    })

    excel_file = model_dir / f"{model_name}_Result.xlsx"

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

    # =====================================
    # SCATTER PLOT
    # =====================================
    plt.figure(figsize=(8, 6))

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

    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    plt.title(
        f"{model_name} (R²={r2:.4f})"
    )

    plt.tight_layout()

    plt.savefig(
        model_dir /
        f"{model_name}_ScatterPlot.png",
        dpi=300
    )

    plt.close()

    # =====================================
    # RESIDUAL PLOT
    # =====================================
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        x=y_pred,
        y=residuals
    )

    plt.axhline(
        y=0,
        color='red',
        linestyle='--'
    )

    plt.xlabel("Predicted")
    plt.ylabel("Residual")

    plt.title(
        f"{model_name} Residual Plot"
    )

    plt.tight_layout()

    plt.savefig(
        model_dir /
        f"{model_name}_ResidualPlot.png",
        dpi=300
    )

    plt.close()

    # =====================================
    # ERROR DISTRIBUTION
    # =====================================
    plt.figure(figsize=(8, 6))

    sns.histplot(
        residuals,
        kde=True
    )

    plt.title(
        f"{model_name} Error Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        model_dir /
        f"{model_name}_ErrorDistribution.png",
        dpi=300
    )

    plt.close()

    # =====================================
    # RETURN METRICS
    # =====================================
    return {
        "Model": model_name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }