import numpy as np
import pandas as pd
file_path ="E:\Hoc_may\Dataset\\"
name_file ="benhnhan.csv"
df = pd.read_csv(file_path + name_file, encoding='windows-1258')
print(df)
print(df.columns)
# ma hoa du lieu thanh so
mapping_dict = {
    'Cannang': {'nhe': 0, 'trungbinh': 1, 'nang': 2},
    'chieucao': {'thap': 0, 'trungbinh': 1, 'cao': 2},
    'huyetap': {'thap': 0, 'trungbinh': 1, 'cao': 2},
    'vandong': {'it': 0, 'trungbinh': 1, 'nhieu': 2},
    'benhtim': {'khong': 0, 'co': 1}
}
# Mapping thủ công
df_new = df.copy()
for col in mapping_dict:
    df_new[col] = df[col].map(mapping_dict[col])

# Hiển thị df_new đã mã hóa
print(df_new)
 #Lấy 8 dòng đầu tiên cho X_train, bỏ cột cuối
X_train = df_new.iloc[:8, :-1]
# Lấy nhãn (label) từ cột cuối cho y_train
y_train = df_new.iloc[:8, -1]
print("X_train:\n", X_train)
print("\ny_train:\n", y_train)
# Lấy 2 dòng cuối cho X_test, bỏ cột cuối (nhãn)
X_test = df_new.iloc[-2:, :-1]
# Nếu bạn cũng cần y_test (nhãn tương ứng)
y_test = df_new.iloc[-2:, -1]
print("X_test:\n", X_test)
print("\ny_test:\n", y_test)

# Nạp mô hình học máy cây quyết định, và học từ dữ liệu trên
from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model)
model.fit(X_train, y_train)
# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

print("kết quả dự đoán cho 2 bệnh nhân X_test:",y_pred)
