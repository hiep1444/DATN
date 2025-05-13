import pandas as pd
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
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
    exit() # Thoát chương trình nếu không tìm thấy file

# Tách feature và label
X = df[['Toán_TB', 'Văn_TB', 'KH_TB']]
y_d1 = df['d1']
y_d2 = df['d2']
y_d3 = df['d3']
y_total = df['diem']

# Hàm tinh chỉnh mô hình RandomForestRegressor
def tune_model(X_train, y_train, model_name):
    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Standardize features
        ('model', RandomForestRegressor(random_state=42)) # RandomForestRegressor model
    ])

    # Define an expanded parameter grid for GridSearchCV
    # Exploring a wider range of values for key hyperparameters
    param_grid = {
        'model__n_estimators': [100, 200, 300, 500], # More trees
        'model__max_depth': [10, 20, 30, None], # Deeper trees or no limit
        'model__min_samples_split': [2, 5, 10, 20], # Varying minimum samples to split
        'model__min_samples_leaf': [1, 2, 4, 8], # Minimum samples required at a leaf node
        'model__max_features': ['sqrt', 'log2', 1.0] # Number of features to consider for the best split
    }
    # GridSearchCV for thorough tuning on the defined grid
    # Increased cv to 10 for more robust evaluation (optional, can increase computation time)
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0) # Set verbose=0 to suppress GridSearchCV progress
    grid_search.fit(X_train, y_train)

    # Print best parameters found by GridSearchCV - KEEP this for the final output
    print(f"Best params (GridSearchCV for {model_name}):", grid_search.best_params_)
    # print(f"--- Kết thúc tinh chỉnh mô hình: {model_name} ---") # Removed intermediate print

    return grid_search.best_estimator_

# Tách tập train/test (80% train, 20% test)
# Note: When using the ALL dataset, this split is still applied.
X_train_d1, X_test_d1, y_d1_train, y_d1_test = train_test_split(X[['Toán_TB']], y_d1, test_size=0.2, random_state=42)
X_train_d2, X_test_d2, y_d2_train, y_d2_test = train_test_split(X[['Văn_TB']], y_d2, test_size=0.2, random_state=42)
X_train_d3, X_test_d3, y_d3_train, y_d3_test = train_test_split(X[['KH_TB']], y_d3, test_size=0.2, random_state=42)
X_train_total, X_test_total, y_total_train, y_total_test = train_test_split(X, y_total, test_size=0.2, random_state=42)

# print("\n--- Bắt đầu huấn luyện các mô hình ---") # Removed intermediate print
# Huấn luyện mô hình cho từng mục tiêu dự đoán
# Passing the model name to the tuning function for better output print
model_d1 = tune_model(X_train_d1, y_d1_train, "Model d1 (Toán)")
model_d2 = tune_model(X_train_d2, y_d2_train, "Model d2 (Văn)")
model_d3 = tune_model(X_train_d3, y_d3_train, "Model d3 (KHTN)")
model_total = tune_model(X_train_total, y_total_train, "Model Tổng điểm")
# print("\n--- Kết thúc huấn luyện các mô hình ---") # Removed intermediate print


# Dự đoán trên tập test
y_d1_pred = model_d1.predict(X_test_d1)
y_d2_pred = model_d2.predict(X_test_d2)
y_d3_pred = model_d3.predict(X_test_d3)
y_total_pred = model_total.predict(X_test_total)

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

    # KEEP these print statements for the final output format
    print("==============================================")
    print(f"ĐÁNH GIÁ MÔ HÌNH: {model_name}")
    print(f"MAE  (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print("==============================================\n")

# Đánh giá các mô hình trên tập test
# print("\n--- Bắt đầu đánh giá các mô hình ---") # Removed intermediate print
evaluate_model(y_d1_test, y_d1_pred, "Model d1 (Toán)")
evaluate_model(y_d2_test, y_d2_pred, "Model d2 (Văn)")
evaluate_model(y_d3_test, y_d3_pred, "Model d3 (KHTN)")
evaluate_model(y_total_test, y_total_pred, "Model Tổng điểm")
# print("--- Kết thúc đánh giá các mô hình ---") # Removed intermediate print

# ===============================================
# Đánh giá độ chính xác trên tập test 20% với sai số ±15 điểm
# KEEP these print statements for the final output format
print("=========== ĐÁNH GIÁ CHÊNH LỆCH ==========")
# Use the original dataframe to sample 20% for this specific evaluation
count = int(len(df) * 0.2)
# Ensure the sample is taken from the correct dataframe (df)
test_data_accuracy = df.sample(count, random_state=42).copy() # Use .copy() to avoid SettingWithCopyWarning
correct_count = 0

for index, row in test_data_accuracy.iterrows():
    # Prepare input features for prediction (ensure it's a DataFrame with correct columns)
    # Use the same feature columns as used for training model_total
    input_feats = pd.DataFrame([[row['Toán_TB'], row['Văn_TB'], row['KH_TB']]], columns=['Toán_TB', 'Văn_TB', 'KH_TB'])

    # Predict total score using the trained total model
    pred_total = model_total.predict(input_feats)[0]

    # Get the actual total score from the 'diem' column
    actual_total = row['diem']

    # Check if the prediction is within ±15 of the actual score
    if abs(pred_total - actual_total) <= 15:
        correct_count += 1

# Calculate accuracy percentage
accuracy = (correct_count / count) * 100

# Print accuracy result - KEEP this for the final output
print(f"Độ chính xác: {accuracy:.2f}%")
print("==============================================")

# Lưu model
output_dir = "../Do_An_Tot_Nghiep_Hiep-test/outputs/RandomForest" # Changed output directory name
# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save each trained model using joblib
try:
    joblib.dump(model_d1, os.path.join(output_dir, "model_d1.pkl"))
    joblib.dump(model_d2, os.path.join(output_dir, "model_d2.pkl"))
    joblib.dump(model_d3, os.path.join(output_dir, "model_d3.pkl"))
    joblib.dump(model_total, os.path.join(output_dir, "model_total.pkl"))
    # KEEP this final print statement
    print(f"\nTất cả mô hình đã được lưu vào thư mục: {output_dir}")
except Exception as e:
    # KEEP this error print statement
    print(f"\nLỗi khi lưu mô hình: {e}")

