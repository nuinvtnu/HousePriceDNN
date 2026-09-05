import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Regression Models
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from save_result import save_regression_results
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
print("Data head:")
print(df.head())
# Feature columns và Target column (Giữ nguyên định dạng Pandas)
X = df.iloc[:, 3:]
y = df.iloc[:, 2]
# =====================================================
# INITIALIZE MODELS
# =====================================================
models = {
    "LinearRegression": LinearRegression(),
    "BayesianRidge": BayesianRidge(),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "KNN": Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=5))]),
    "SVR": Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf'))]),
    "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
}
# =====================================================
# 5-FOLD CROSS VALIDATION
# =====================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_results = []
for model_name, model in models.items():
    print(f"\nRunning 5-Fold CV for {model_name} ...")
    fold_metrics_list = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_test_fold)
        metrics_fold = save_regression_results(
            model=model,
            model_name=f"{model_name}_Fold{fold + 1}",
            X_test=X_test_fold,
            y_test=y_test_fold,
            y_pred=preds
        )
        if metrics_fold is not None:
            fold_metrics_list.append(metrics_fold)
        print(f"  - Fold {fold + 1} finished.")
    if fold_metrics_list:
        df_fold_metrics = pd.DataFrame(fold_metrics_list)
        mean_metrics = df_fold_metrics.mean(numeric_only=True).to_dict()
        mean_metrics["Model_Name"] = model_name
        all_results.append(mean_metrics)
# =====================================================
# SUMMARY TABLE
# =====================================================
summary_df = pd.DataFrame(all_results)
if not summary_df.empty and "Model_Name" in summary_df.columns:
    cols = ['Model_Name'] + [col for col in summary_df.columns if col != 'Model_Name']
    summary_df = summary_df[cols]
summary_file = RESULT_DIR / "Summary_Regression_5Fold_Average.xlsx"
summary_df.to_excel(summary_file, index=False)
print("\n======================================")
print("ALL MODELS AND FOLDS FINISHED")
print("======================================")
print(summary_df)
print(f"\nSummary saved to: {summary_file}")