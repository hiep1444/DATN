import pandas as pd
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, '..', 'data', 'HSA_HD_Final_Cleaned_ALL_Fixed.csv')

# Đọc dữ liệu bằng đường dẫn đầy đủ đã xây dựng
try:
    df = pd.read_csv(file_path)
    print(f"Đã đọc thành công file: {file_path}")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại đường dẫn: {file_path}")
    print("Vui lòng kiểm tra lại đường dẫn file và cấu trúc thư mục.")
    exit()

# Tách feature và label
X = df[['Toán_TB', 'Văn_TB', 'KH_TB']]
y_d1 = df['d1']  # Toán học
y_d2 = df['d2']  # Văn học
y_d3 = df['d3']  # Khoa học tự nhiên
y_total = df['diem'] # Total score

# Hàm tinh chỉnh mô hình LinearRegression
def tune_model(X_train, y_train, model_name):
    """
    Tunes a LinearRegression model using GridSearchCV.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        model_name (str): Name of the model being tuned (for printing).

    Returns:
        sklearn.pipeline.Pipeline: The best trained model pipeline.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Standardize features
        ('model', LinearRegression()) # Linear Regression model
    ])

    param_grid = {
        'model__fit_intercept': [True, False], # Whether to calculate the intercept for this model
        'model__positive': [True, False] # When set to True, forces the coefficients to be positive
    }

    # GridSearchCV for tuning on the defined grid
    # verbose=0 to suppress GridSearchCV progress
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
    print(f"Bắt đầu tinh chỉnh mô hình: {model_name}...")
    grid_search.fit(X_train, y_train)
    print(f"Tinh chỉnh hoàn tất cho: {model_name}.")

    # Print best parameters found by GridSearchCV - KEEP this for the final output
    print(f"Best params (GridSearchCV for {model_name}):", grid_search.best_params_)

    return grid_search.best_estimator_

# Tách tập train/test (80% train, 20% test)
# Split data for individual scores (using single feature)
X_train_d1, X_test_d1, y_d1_train, y_d1_test = train_test_split(X[['Toán_TB']], y_d1, test_size=0.2, random_state=42)
X_train_d2, X_test_d2, y_d2_train, y_d2_test = train_test_split(X[['Văn_TB']], y_d2, test_size=0.2, random_state=42)
X_train_d3, X_test_d3, y_d3_train, y_d3_test = train_test_split(X[['KH_TB']], y_d3, test_size=0.2, random_state=42)

# Split data for total score (using all features)
X_train_total, X_test_total, y_total_train, y_total_test = train_test_split(X, y_total, test_size=0.2, random_state=42)


# Huấn luyện mô hình
# Train models for individual scores
model_d1 = tune_model(X_train_d1, y_d1_train, "Model d1 (Toán)")
model_d2 = tune_model(X_train_d2, y_d2_train, "Model d2 (Văn)")
model_d3 = tune_model(X_train_d3, y_d3_train, "Model d3 (KHTN)")
# Train a separate model for the total score
model_total = tune_model(X_train_total, y_total_train, "Model Tổng điểm")


# Dự đoán trên tập test
print("\nTiến hành dự đoán trên tập test...")
y_d1_pred = model_d1.predict(X_test_d1)
y_d2_pred = model_d2.predict(X_test_d2)
y_d3_pred = model_d3.predict(X_test_d3)
# Predict total score using the dedicated total model
y_total_pred = model_total.predict(X_test_total)
print("Dự đoán hoàn tất.")


# Hàm đánh giá mô hình và in kết quả theo format yêu cầu
def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluates the model performance using MAE and RMSE and prints results in the specified format.

    Args:
        y_true (pd.Series or np.array): True labels.
        y_pred (np.array): Predicted labels.
        model_name (str): Name of the model being evaluated.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Đánh giá các mô hình
print("\n=========== ĐÁNH GIÁ MÔ HÌNH ==========")
evaluate_model(y_d1_test, y_d1_pred, "Model d1")
evaluate_model(y_d2_test, y_d2_pred, "Model d2")
evaluate_model(y_d3_test, y_d3_pred, "Model d3")
# --- Thêm đánh giá cho mô hình tổng điểm ---
evaluate_model(y_total_test, y_total_pred, "Model Tổng điểm")
# -----------------------------------------
print("==============================================")


# Đánh giá độ chính xác trên tập test 20% với sai số ±15 điểm
# Để đánh giá trên tập test đã chia, bạn nên sử dụng X_test_total và y_total_test
print("\n=========== ĐÁNH GIÁ CHÊNH LỆCH TRÊN TẬP TEST ==========")
# Sử dụng tập test đã chia để đánh giá độ lệch
test_data_for_accuracy = pd.DataFrame(X_test_total, columns=['Toán_TB', 'Văn_TB', 'KH_TB'])
test_data_for_accuracy['actual_diem'] = y_total_test.values # Thêm cột điểm thực tế vào DataFrame test

correct_count = 0
tolerance = 15 # Sai số cho phép

# Dự đoán lại trên tập test để so sánh
# pred_total_on_test đã được tính ở trên: y_total_pred

for index, row in test_data_for_accuracy.iterrows():
    actual_total = row['actual_diem']
    # Lấy giá trị dự đoán tương ứng từ mảng y_total_pred
    # Sử dụng index của row trong test_data_for_accuracy để tìm vị trí tương ứng trong X_test_total (và y_total_pred)
    # Cách an toàn hơn là dùng index của X_test_total
    original_index = X_test_total.index[test_data_for_accuracy.index.get_loc(index)]
    pred_total = y_total_pred[X_test_total.index.get_loc(original_index)]


    # Check if the prediction is within ±15 of the actual score
    if abs(pred_total - actual_total) <= tolerance:
        correct_count += 1

# Calculate accuracy percentage
accuracy = (correct_count / len(test_data_for_accuracy)) * 100

# Print accuracy result - KEEP this for the final output
print(f"Độ chính xác: {accuracy:.2f}%")
print("==============================================")

# Lưu model
output_dir = "../Do_An_Tot_Nghiep_Hiep-test/outputs/LinearRegression" # Changed output directory name
# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save each trained model using joblib
try:
    joblib.dump(model_d1, os.path.join(output_dir, "model_d1.pkl"))
    joblib.dump(model_d2, os.path.join(output_dir, "model_d2.pkl"))
    joblib.dump(model_d3, os.path.join(output_dir, "model_d3.pkl"))
    joblib.dump(model_total, os.path.join(output_dir, "model_total.pkl")) # Save the total model
    # KEEP this final print statement
    print(f"\nTất cả mô hình đã được lưu vào thư mục: {output_dir}")
except Exception as e:
    # KEEP this error print statement
    print(f"\nLỗi khi lưu mô hình: {e}")
