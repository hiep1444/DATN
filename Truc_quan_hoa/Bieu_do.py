import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc file CSV
df = pd.read_csv("data/HSA_HD_Final.csv")


# Các cột điểm tổng kết CN của 3 năm học
grade_cols = ['10.Điểm tổng kết CN', '11.Điểm tổng kết CN', '12.Điểm tổng kết CN']

# Chuyển đổi dữ liệu về kiểu số an toàn
for col in grade_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 1. Boxplot: So sánh điểm tổng kết CN giữa các năm học
plt.figure(figsize=(10, 6))
df_melted = df[grade_cols].melt(var_name='Năm học', value_name='Điểm tổng kết')
sns.boxplot(x='Năm học', y='Điểm tổng kết', data=df_melted)
plt.title('So sánh điểm tổng kết CN giữa các năm học')
plt.xlabel('Năm học')
plt.ylabel('Điểm tổng kết')
plt.show()

# 2. Barplot: Điểm trung bình tổng kết CN theo năm học
avg_scores = df[grade_cols].mean().reset_index()
avg_scores.columns = ['Năm học', 'Điểm trung bình']

plt.figure(figsize=(8, 5))
sns.barplot(x='Năm học', y='Điểm trung bình', data=avg_scores)
plt.title('Điểm trung bình tổng kết CN theo năm học')
plt.ylabel('Điểm trung bình')
plt.xlabel('Năm học')
plt.ylim(0, 10)
plt.show()

# 3. Vẽ biểu đồ phân phối điểm tổng kết CN riêng biệt cho từng lớp
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, col in enumerate(grade_cols):
    sns.histplot(df[col].dropna(), kde=True, bins=30, stat='density', ax=axes[i])
    axes[i].set_title(f'Phân phối điểm tổng kết CN lớp {col[:2]}')
    axes[i].set_xlabel('Điểm tổng kết')
    axes[i].set_ylabel('Mật độ')

plt.tight_layout()
plt.show()

# 4.Tạo biểu đồ cột nhóm cho xu hướng điểm trung bình theo học kỳ

# Chọn các cột điểm tổng kết HK I và HK II từ lớp 10 đến 12
semesters = ['HK I', 'HK II']
grades = ['10', '11', '12']
semester_cols = [f"{g}.Điểm tổng kết {s}" for g in grades for s in semesters]

# Chuyển đổi dữ liệu về kiểu số
for col in semester_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Tính điểm trung bình mỗi học kỳ
avg_semester_scores = pd.DataFrame({
    'Lớp': [],
    'Học kỳ': [],
    'Điểm trung bình': []
})

for g in grades:
    for s in semesters:
        col = f"{g}.Điểm tổng kết {s}"
        avg = df[col].mean()
        avg_semester_scores = avg_semester_scores.append({
            'Lớp': g,
            'Học kỳ': s,
            'Điểm trung bình': avg
        }, ignore_index=True)

# Vẽ biểu đồ cột nhóm
plt.figure(figsize=(10, 6))
sns.barplot(x='Lớp', y='Điểm trung bình', hue='Học kỳ', data=avg_semester_scores)
plt.title('Xu hướng điểm trung bình theo học kỳ và lớp')
plt.ylabel('Điểm trung bình')
plt.xlabel('Lớp')
plt.ylim(0, 10)
plt.legend(title='Học kỳ')
plt.tight_layout()
plt.show()
