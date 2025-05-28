import pandas as pd
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, '..', 'data', 'HSA_HD_Final_Cleaned_All_Fixed.csv')

# Đọc dữ liệu bằng đường dẫn đầy đủ đã xây dựng
try:
    df = pd.read_csv(file_path)
    print(f"Đã đọc thành công file: {file_path}")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại đường dẫn: {file_path}")
    print("Vui lòng kiểm tra lại đường dẫn file và cấu trúc thư mục.")
    exit() # Thoát chương trình nếu không tìm thấy file

# Tách feature và label
X = df[['Toán_TB', 'Văn_TB', 'KH_TB']]
y_d1 = df['d1']
y_d2 = df['d2']
y_d3 = df['d3']
y_total = df['diem']

# Hàm tinh chỉnh mô hình SVR
def tune_model(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR())
    ])

    param_grid = {
        'model__C': [1, 10, 100],
        'model__gamma': ['scale', 0.1, 1],
        'model__kernel': ['rbf']
    }

    # Sử dụng scoring là 'neg_mean_squared_error' và n_jobs=-1 để tận dụng CPU
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    print("Bắt đầu tinh chỉnh mô hình...")
    grid_search.fit(X_train, y_train)
    print("Tinh chỉnh hoàn tất.")
    print("Best params:", grid_search.best_params_)
    return grid_search.best_estimator_

# Tách tập train/test
# Sử dụng toàn bộ X cho tập total, và các cột riêng lẻ cho d1, d2, d3
X_train_d1, X_test_d1, y_d1_train, y_d1_test = train_test_split(X[['Toán_TB']], y_d1, test_size=0.2, random_state=42)
X_train_d2, X_test_d2, y_d2_train, y_d2_test = train_test_split(X[['Văn_TB']], y_d2, test_size=0.2, random_state=42)
X_train_d3, X_test_d3, y_d3_train, y_d3_test = train_test_split(X[['KH_TB']], y_d3, test_size=0.2, random_state=42)
X_train_total, X_test_total, y_total_train, y_total_test = train_test_split(X, y_total, test_size=0.2, random_state=42)

# Huấn luyện mô hình
print("\nHuấn luyện và tinh chỉnh Model d1 (Toán)...")
model_d1 = tune_model(X_train_d1, y_d1_train)
print("\nHuấn luyện và tinh chỉnh Model d2 (Văn)...")
model_d2 = tune_model(X_train_d2, y_d2_train)
print("\nHuấn luyện và tinh chỉnh Model d3 (KHTN)...")
model_d3 = tune_model(X_train_d3, y_d3_train)
print("\nHuấn luyện và tinh chỉnh Model Tổng điểm...")
model_total = tune_model(X_train_total, y_total_train)

# Dự đoán
print("\nTiến hành dự đoán trên tập test...")
y_d1_pred = model_d1.predict(X_test_d1)
y_d2_pred = model_d2.predict(X_test_d2)
y_d3_pred = model_d3.predict(X_test_d3)
y_total_pred = model_total.predict(X_test_total)
print("Dự đoán hoàn tất.")

# Hàm đánh giá mô hình
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print("==============================================")
    print(f"ĐÁNH GIÁ MÔ HÌNH: {model_name}")
    print(f"MAE  (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print("==============================================\n")

# Đánh giá các mô hình
evaluate_model(y_d1_test, y_d1_pred, "Model d1 (Toán)")
evaluate_model(y_d2_test, y_d2_pred, "Model d2 (Văn)")
evaluate_model(y_d3_test, y_d3_pred, "Model d3 (KHTN)")
evaluate_model(y_total_test, y_total_pred, "Model Tổng điểm")

# Đánh giá độ chính xác trên tập test 20% với sai số ±15 điểm
# Lưu ý: Phần này đang lấy mẫu ngẫu nhiên từ toàn bộ df, không phải từ tập test đã chia ở trên.
# Để đánh giá trên tập test đã chia, bạn nên sử dụng X_test_total và y_total_test
print("=========== ĐÁNH GIÁ CHÊNH LỆCH TRÊN TẬP TEST ==========")
# Sử dụng tập test đã chia để đánh giá độ lệch
test_data_for_accuracy = pd.DataFrame(X_test_total, columns=['Toán_TB', 'Văn_TB', 'KH_TB'])
test_data_for_accuracy['actual_diem'] = y_total_test.values # Thêm cột điểm thực tế vào DataFrame test

correct_count = 0
tolerance = 15 # Sai số cho phép

# Dự đoán lại trên tập test để so sánh
pred_total_on_test = model_total.predict(X_test_total)

for index, row in test_data_for_accuracy.iterrows():
    actual_total = row['actual_diem']
    # Lấy giá trị dự đoán tương ứng từ mảng pred_total_on_test
    pred_total = pred_total_on_test[test_data_for_accuracy.index.get_loc(index)]

    if abs(pred_total - actual_total) <= tolerance:
        correct_count += 1

accuracy = correct_count / len(test_data_for_accuracy) * 100
print(f"Độ chính xác): {accuracy:.2f}%")
print("==============================================")


# Lưu model
# --- Sửa đổi cách lưu model để không phụ thuộc vào working directory ---
output_dir = "../Do_An_Tot_Nghiep_Hiep-test/outputs/SVM" # Changed output directory name
# ---------------------------------------------------------------------
os.makedirs(output_dir, exist_ok=True)

joblib.dump(model_d1, os.path.join(output_dir, "model_d1.pkl"))
joblib.dump(model_d2, os.path.join(output_dir, "model_d2.pkl"))
joblib.dump(model_d3, os.path.join(output_dir, "model_d3.pkl"))
joblib.dump(model_total, os.path.join(output_dir, "model_total.pkl"))

print(f"\nTất cả mô hình SVM đã được lưu vào thư mục: {output_dir}")
