# main_regression.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Regression Models
from sklearn.linear_model import (
LinearRegression,
BayesianRidge
)
from sklearn.ensemble import (RandomForestRegressor)
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
print(df.head())
# Feature columns
X = df.iloc[:, 3:]
# Target column (price)
y = df.iloc[:, 2]
# =====================================================
# TRAIN TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
X,y,test_size=0.20,random_state=42)
# =====================================================
# STANDARDIZE DATA
# =====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# =====================================================
# STORE RESULTS
# ====================================================
all_results = []
# =====================================================
# 1. LINEAR REGRESSION
# =====================================================
print("\nRunning Linear Regression ...")
model_lr = LinearRegression()
model_lr.fit(X_train,y_train)
y_pred = model_lr.predict(X_test)
metrics_lr = save_regression_results(
model=model_lr,
model_name="LinearRegression",
X_test=X_test,
y_test=y_test,
y_pred=y_pred
)
all_results.append(metrics_lr)
# =====================================================
# 2. BAYESIAN RIDGE
# =====================================================
print("\nRunning Bayesian Ridge ...")
model_bayes = BayesianRidge()
model_bayes.fit(X_train,y_train)
y_pred = model_bayes.predict(X_test)
metrics_bayes = save_regression_results(
model=model_bayes,
model_name="BayesianRidge",
X_test=X_test,
y_test=y_test,
y_pred=y_pred
)
all_results.append(metrics_bayes)
# =====================================================
# 3. RANDOM FOREST
# =====================================================
print("\nRunning Random Forest ...")
model_rf = RandomForestRegressor(
n_estimators=300,
random_state=42,
n_jobs=-1
)
model_rf.fit(X_train,y_train)
y_pred = model_rf.predict(X_test)
metrics_rf = save_regression_results(
model=model_rf,
model_name="RandomForest",
X_test=X_test,
y_test=y_test,
y_pred=y_pred
)
all_results.append(metrics_rf)
# =====================================================
# 4. KNN
# =====================================================
print("\nRunning KNN ...")
model_knn = KNeighborsRegressor(n_neighbors=5)
model_knn.fit(X_train_scaled,y_train)
y_pred = model_knn.predict(X_test_scaled)
metrics_knn = save_regression_results(
model=model_knn,
model_name="KNN",
X_test=X_test,
y_test=y_test,
y_pred=y_pred
)
all_results.append(metrics_knn)
# =====================================================
# 5. SVR
# =====================================================
print("\nRunning SVR ...")
model_svr = SVR(
kernel='rbf'
)
model_svr.fit(
X_train_scaled,
y_train
)
y_pred = model_svr.predict(
X_test_scaled
)
metrics_svr = save_regression_results(
model=model_svr,
model_name="SVR",
X_test=X_test,
y_test=y_test,
y_pred=y_pred
)
all_results.append(metrics_svr)
# =====================================================
# 6. XGBOOST
# =====================================================
print("\nRunning XGBoost ...")
model_xgb = XGBRegressor(
n_estimators=300,
max_depth=6,
learning_rate=0.05,
random_state=42
)
model_xgb.fit(
X_train,
y_train
)
y_pred = model_xgb.predict(X_test)
metrics_xgb = save_regression_results(
model=model_xgb,
model_name="XGBoost",
X_test=X_test,
y_test=y_test,
y_pred=y_pred
)
all_results.append(metrics_xgb)
# =====================================================
# SUMMARY TABLE
# =====================================================
summary_df = pd.DataFrame(all_results)
summary_file = (RESULT_DIR /"Summary_Regression.xlsx")
summary_df.to_excel(summary_file,index=False)
print("\n======================================")
print("ALL MODELS FINISHED")
print("======================================")
print(summary_df)
print(f"\nSummary saved to: {summary_file}")
