import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Khởi tạo dữ liệu từ bảng kết quả mới của bạn
data = {
    "Feature": [
        "lat", "long", "grade", "sqft_living", "sqft_above", "zipcode",
        "waterfront", "sqft_living15", "yr_built", "floors", "bathrooms",
        "condition", "yr_renovated", "bedrooms", "sqft_basement", "sqft_lot",
        "view", "sqft_lot15"
    ],
    "Importance_Drop_R2": [
        0.392335, 0.273605, 0.126633, 0.110281, 0.075859, 0.041908,
        0.041656, 0.040439, 0.024491, 0.018706, 0.016809, 0.008657,
        0.007518, 0.007104, 0.006466, 0.004425, 0.004217, 0.003558
    ],
    "Std_Dev": [
        0.012143, 0.019523, 0.008592, 0.004327, 0.003098, 0.007373,
        0.002368, 0.005192, 0.001354, 0.003434, 0.001509, 0.000708,
        0.000527, 0.001541, 0.001718, 0.002032, 0.000603, 0.002440
    ]
}

df_plot = pd.DataFrame(data)

# 2. Cấu hình giao diện đồ thị (Tăng chiều cao figure lên một chút để thoáng không gian)
plt.figure(figsize=(11, 8.5))
sns.set_theme(style="whitegrid")

# 3. Vẽ đồ thị thanh ngang với bảng màu chuyên nghiệp
ax = sns.barplot(
    x="Importance_Drop_R2",
    y="Feature",
    data=df_plot,
    palette="viridis",
    edgecolor="black",
    linewidth=0.7
)

# Thêm thanh sai số (Standard Deviation)
ax.errorbar(
    x=df_plot["Importance_Drop_R2"],
    y=df_plot["Feature"],
    xerr=df_plot["Std_Dev"],
    fmt='none',
    c='black',
    capsize=3,
    label='Standard Deviation (Std Dev)'
)

# 4. Hiển thị giá trị số cụ thể ở cuối mỗi thanh
for i, val in enumerate(df_plot["Importance_Drop_R2"]):
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

# 5. Tiêu đề và nhãn trục bằng tiếng Anh chuẩn học thuật
# Hạ nhỏ size từ 14 xuống 13 và chỉnh lại khoảng đệm pad=20 để đẩy đồ thị xuống dưới
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

# Thêm chú thích cho thanh sai số ở góc phải dưới
plt.legend(loc="lower right")

# 6. SỬA LỖI CHE KHUẤT: Thiết lập khoảng cách lề thủ công để ép tiêu đề không chạm viền
plt.subplots_adjust(top=0.88, bottom=0.12, left=0.15, right=0.95)

# Lưu/xuất file hình ảnh chất lượng cao
plt.savefig("DNN_FeatureImportance_Visual_EN.png", dpi=300)
plt.show()