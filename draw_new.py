import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. CẬP NHẬT DỮ LIỆU MỚI (Đã sắp xếp theo thứ tự giảm dần của Importance_Drop_R2)
data = {
    "Feature": [
        "lat",
        "long",
        "sqft_living",
        "grade",
        "sqft_above",
        "zipcode",
        "waterfront",
        "sqft_living15",
        "yr_built",
        "floors",
        "bathrooms",
        "yr_renovated",
        "condition",
        "bedrooms",
        "sqft_basement",
        "sqft_lot15",
        "sqft_lot",
        "view",
    ],
    "Importance_Drop_R2": [
        0.4028,
        0.2826,
        0.1370,
        0.1172,
        0.0499,
        0.0449,
        0.0375,
        0.0342,
        0.0228,
        0.0224,
        0.0152,
        0.0079,
        0.0078,
        0.0068,
        0.0065,
        0.0060,
        0.0052,
        0.0042,
    ],
    "Std_Dev": [
        0.0129,
        0.0168,
        0.0045,
        0.0074,
        0.0031,
        0.0071,
        0.0028,
        0.0031,
        0.0018,
        0.0028,
        0.0013,
        0.0013,
        0.0011,
        0.0008,
        0.0016,
        0.0027,
        0.0014,
        0.0018,
    ],
}

# Đảo ngược bảng dữ liệu để tính năng cao nhất nằm ở TRÊN CÙNG khi vẽ đồ thị thanh ngang
df_plot = pd.DataFrame(data).iloc[::-1].reset_index(drop=True)

# 2. Cấu hình giao diện đồ thị
plt.figure(figsize=(11, 8.5))
sns.set_theme(style="whitegrid")

# Tạo bảng màu viridis tương ứng với số lượng feature (18 màu)
colors = sns.color_palette("viridis", len(df_plot))

# 3. Vẽ đồ thị thanh ngang
bars = plt.barh(
    df_plot["Feature"],
    df_plot["Importance_Drop_R2"],
    color=colors,
    edgecolor="black",
    linewidth=0.7,
    height=0.6,
)

# Thêm thanh sai số (Standard Deviation) chuẩn xác theo từng tọa độ y
plt.errorbar(
    x=df_plot["Importance_Drop_R2"],
    y=df_plot["Feature"],
    xerr=df_plot["Std_Dev"],
    fmt="none",
    c="black",
    capsize=3,
    label="Standard Deviation (Std Dev)",
)

# 4. Hiển thị giá trị số cụ thể ở cuối mỗi thanh
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.4f}",
        va="center",
        ha="left",
        fontsize=10,
        weight="bold",
    )

# 5. Tiêu đề và nhãn trục bằng tiếng Anh chuẩn học thuật
plt.title(
    "FEATURE IMPORTANCE RANKING FOR HOUSE PRICE PREDICTION\n(Permutation Importance Method on DNN Model)",
    fontsize=13,
    weight="bold",
    pad=20,
)
plt.xlabel(
    "Drop in R² Score upon Feature Permutation (Higher is more important)",
    fontsize=12,
    labelpad=10,
)
plt.ylabel("House Features", fontsize=12)

# Giới hạn trục X rộng thêm một chút để không bị che khuất nhãn số của thanh cao nhất (lat)
plt.xlim(0, df_plot["Importance_Drop_R2"].max() + 0.05)

# Thêm chú thích cho thanh sai số ở góc phải dưới
plt.legend(loc="lower right")

# 6. Thiết lập khoảng cách lề thủ công để ép tiêu đề không chạm viền
plt.subplots_adjust(top=0.88, bottom=0.12, left=0.15, right=0.95)

# Lưu/xuất file hình ảnh chất lượng cao với số liệu mới
plt.savefig("DNN_FeatureImportance_Visual_EN_Updated.png", dpi=300)
plt.show()