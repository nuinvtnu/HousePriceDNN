from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as RMSE
from sklearn.metrics import r2_score as R2
from sklearn.svm import SVR


file_path ="E:\Hoc_may\Dataset\\"
path_result = "E:\Hoc_may\Resultx\\"
name_file ="kc_house_data.csv"
df = pd.read_csv(file_path + name_file, encoding='windows-1258')
print(df.head())
X= df.iloc[:,3:]# clumn feature from 3-final
print(X[:5])
Y = df.iloc[:,2] # get all recode in column 2 ( prize is 2 (column in df begin 0)
print(Y[:5])
X_train, X_test,y_train,y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = SVR()
model.fit(X_train, y_train)
model.score(X_test, y_test)
y_pred = model.predict(X_test)
print("MSE:", MSE(y_test, y_pred))
print("RMSE:", RMSE(y_test, y_pred))
print("r2:", R2(y_test, y_pred))
# Tạo DataFrame kết quả: gồm X_test, y_test, y_pred
results_df = X_test.copy()  # copy X_test để tránh thay đổi bản gốc
results_df['y_test'] = y_test.values
results_df['y_pred'] = y_pred

# Ghi vào file Excel
output_file = path_result+ "SVR.xlsx"
results_df.to_excel(output_file, index=False)

print(f"✅ Đã lưu X_test, y_test và y_pred vào: {output_file}")
# vẽ r2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# Tính r2 lại (nếu chưa)
r2 = r2_score(y_test, y_pred)

# Tạo biểu đồ
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.5, s=40)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='y = y')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title(f'Scatter plot: y_test vs y_pred (R² = {r2:.3f})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
