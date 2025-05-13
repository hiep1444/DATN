import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc file CSV
df = pd.read_csv("data/HSA_HD_Final.csv")

# Tạo cột điểm tổng kết KHXH (trung bình 3 môn Sử, Địa, GDCD)
df["KHXH_12"] = df[["12.Lịch sử CN", "12.Địa lí CN", "12.GDCD CN"]].mean(axis=1)

# 1. Biểu đồ điểm trung bình theo từng lớp với hiệu ứng
plt.figure(figsize=(10, 6))
colors = ["blue", "green", "red"]
means = df[["10.Điểm tổng kết CN", "11.Điểm tổng kết CN", "12.Điểm tổng kết CN"]].mean()
sns.barplot(x=["Lớp 10", "Lớp 11", "Lớp 12"], y=means, palette=colors)
for i, v in enumerate(means):
    plt.text(i, v + 0.1, f"{v:.2f}", ha='center', fontsize=12, fontweight='bold')
plt.xlabel("Lớp học")
plt.ylabel("Điểm trung bình")
plt.title("Xu hướng điểm trung bình qua các lớp", fontsize=14, fontweight='bold')
plt.show()

#2 Biểu đồ tương quan giữa điểm thi HSA với các đầu điểm toán, văn, KHXH
fig, axes = plt.subplots(3, 1, figsize=(6, 12))

sns.regplot(x=df["d1"], y=df["12.Toán CN"], ax=axes[0], scatter_kws={'s': 10}, line_kws={'color': 'red'})
axes[0].set_xlabel("Điểm thi HSA - Toán (d1)")
axes[0].set_ylabel("Điểm tổng kết Toán lớp 12")
axes[0].set_title("Mối tương quan giữa điểm thi HSA và điểm tổng kết")

sns.regplot(x=df["d2"], y=df["12.Văn CN"], ax=axes[1], scatter_kws={'s': 10}, line_kws={'color': 'red'})
axes[1].set_xlabel("Điểm thi HSA - Văn (d2)")
axes[1].set_ylabel("Điểm tổng kết Văn lớp 12")

sns.regplot(x=df["d3"], y=df["KHXH_12"], ax=axes[2], scatter_kws={'s': 10}, line_kws={'color': 'red'})
axes[2].set_xlabel("Điểm thi HSA - KHXH (d3)")
axes[2].set_ylabel("Điểm tổng kết KHXH lớp 12")

plt.tight_layout()
plt.show()


# 3. Phân bố điểm qua các lớp, tách riêng từng lớp
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

sns.histplot(df["10.Điểm tổng kết CN"], bins=20, kde=True, color="blue", ax=axes[0])
axes[0].set_title("Phân bố điểm lớp 10")
axes[0].set_xlabel("Điểm trung bình")
axes[0].set_ylabel("Số học sinh")

sns.histplot(df["11.Điểm tổng kết CN"], bins=20, kde=True, color="green", ax=axes[1])
axes[1].set_title("Phân bố điểm lớp 11")
axes[1].set_xlabel("Điểm trung bình")
axes[1].set_ylabel("Số học sinh")

sns.histplot(df["12.Điểm tổng kết CN"], bins=20, kde=True, color="red", ax=axes[2])
axes[2].set_title("Phân bố điểm lớp 12")
axes[2].set_xlabel("Điểm trung bình")
axes[2].set_ylabel("Số học sinh")

plt.tight_layout()
plt.show()

# 4. Biểu đồ xu hướng điểm trung bình theo học kỳ - Đổi thành biểu đồ cột nhóm
plt.figure(figsize=(10, 6))
x_labels = ["10.Điểm tổng kết HK I", "10.Điểm tổng kết HK II", "11.Điểm tổng kết HK I", "11.Điểm tổng kết HK II", "12.Điểm tổng kết HK I", "12.Điểm tổng kết HK II"]
y_values = [
    df["10.Điểm tổng kết HK I"].mean(), df["10.Điểm tổng kết HK II"].mean(),
    df["11.Điểm tổng kết HK I"].mean(), df["11.Điểm tổng kết HK II"].mean(),
    df["12.Điểm tổng kết HK I"].mean(), df["12.Điểm tổng kết HK II"].mean()
]
sns.barplot(x=x_labels, y=y_values, palette="coolwarm")
plt.xlabel("Học kỳ")
plt.ylabel("Điểm trung bình")
plt.title("Xu hướng điểm trung bình theo học kỳ (Biểu đồ cột)")
plt.xticks(rotation=45)
plt.show()

# 5. Biểu đồ mật độ điểm tổng kết cả 3 lớp
plt.figure(figsize=(8, 5))
sns.kdeplot(df["10.Điểm tổng kết CN"], color="blue", fill=True, label="Lớp 10")
sns.kdeplot(df["11.Điểm tổng kết CN"], color="green", fill=True, label="Lớp 11")
sns.kdeplot(df["12.Điểm tổng kết CN"], color="red", fill=True, label="Lớp 12")
plt.xlabel("Điểm trung bình")
plt.ylabel("Mật độ")
plt.title("Mật độ điểm tổng kết của từng lớp")
plt.legend()
plt.show()