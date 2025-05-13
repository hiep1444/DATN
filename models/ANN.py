import pandas as pd
import joblib
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
y_d1 = df['d1']  # Toán học
y_d2 = df['d2']  # Văn học
y_d3 = df['d3']  # Khoa học tự nhiên
y_total = df['diem'] # Total score

# Tách tập train/test (80% train, 20% test)
# Split X (all features) for all models
X_train, X_test, y_d1_train, y_d1_test = train_test_split(X, y_d1, test_size=0.2, random_state=42)
# Split y for other targets, ensuring consistency with X_train/X_test split
_, _, y_d2_train, y_d2_test = train_test_split(X, y_d2, test_size=0.2, random_state=42)
_, _, y_d3_train, y_d3_test = train_test_split(X, y_d3, test_size=0.2, random_state=42)
# Split y for total score, ensuring consistency with X_train/X_test split
_, _, y_total_train, y_total_test = train_test_split(X, y_total, test_size=0.2, random_state=42)


# Chuẩn hóa dữ liệu
# Fit scaler on the training data (all features) and transform both train and test sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hàm tạo mô hình ANN
def build_ann_model():
    """
    Builds a Sequential Keras model for regression.
    """
    model = keras.Sequential([
        # Input layer: 3 features
        keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)), # Use shape from scaled data
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3), # Dropout for regularization
        keras.layers.Dense(32, activation='relu'),
        # Output layer: 1 neuron for regression output
        keras.layers.Dense(1)
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# Huấn luyện mô hình
print("--- Bắt đầu huấn luyện các mô hình ANN ---")
print("Huấn luyện Model d1 (Toán)...")
model_d1 = build_ann_model()
# Use scaled training data for fitting
history_d1 = model_d1.fit(X_train_scaled, y_d1_train, epochs=500, batch_size=16, verbose=0, validation_split=0.1, callbacks=[early_stop])

print("Huấn luyện Model d2 (Văn)...")
model_d2 = build_ann_model()
# Use scaled training data for fitting
history_d2 = model_d2.fit(X_train_scaled, y_d2_train, epochs=500, batch_size=16, verbose=0, validation_split=0.1, callbacks=[early_stop])

print("Huấn luyện Model d3 (KHTN)...")
model_d3 = build_ann_model()
# Use scaled training data for fitting
history_d3 = model_d3.fit(X_train_scaled, y_d3_train, epochs=500, batch_size=16, verbose=0, validation_split=0.1, callbacks=[early_stop])

# Train a separate model for the total score using all features
print("Huấn luyện Model Tổng điểm...")
model_total = build_ann_model()
# Use scaled training data for fitting the total model
history_total = model_total.fit(X_train_scaled, y_total_train, epochs=500, batch_size=16, verbose=0, validation_split=0.1, callbacks=[early_stop])
print("--- Kết thúc huấn luyện các mô hình ANN ---")


# Dự đoán trên tập test
# Use scaled test data for prediction
y_d1_pred = model_d1.predict(X_test_scaled).flatten()
y_d2_pred = model_d2.predict(X_test_scaled).flatten()
y_d3_pred = model_d3.predict(X_test_scaled).flatten()
# Predict total score using the dedicated total model on scaled test data
y_total_pred = model_total.predict(X_test_scaled).flatten()


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
print("--- Bắt đầu đánh giá các mô hình ---")
evaluate_model(y_d1_test, y_d1_pred, "Model d1 (Toán)")
evaluate_model(y_d2_test, y_d2_pred, "Model d2 (Văn)")
evaluate_model(y_d3_test, y_d3_pred, "Model d3 (KHTN)")
evaluate_model(y_total_test, y_total_pred, "Model Tổng điểm") 
print("--- Kết thúc đánh giá các mô hình ---")

print("=========== ĐÁNH GIÁ CHÊNH LỆCH ==========")
count = int(len(df) * 0.2)
test_data_accuracy = df.sample(count, random_state=42).copy() 
correct_count = 0

for index, row in test_data_accuracy.iterrows():
    input_features = pd.DataFrame([[row['Toán_TB'], row['Văn_TB'], row['KH_TB']]], columns=['Toán_TB', 'Văn_TB', 'KH_TB'])
    input_features_scaled = scaler.transform(input_features)
    pred_total = model_total.predict(input_features_scaled)[0][0]
    # Get the actual total score from the 'diem' column
    actual_total = row['diem']

    # Check if the prediction is within ±15 of the actual score
    if abs(pred_total - actual_total) <= 15:
        correct_count += 1

accuracy = (correct_count / count) * 100

print(f"Độ chính xác: {accuracy:.2f}%")
print("==============================================")

# Lưu model và scaler
output_dir = "../Do_An_Tot_Nghiep_Hiep-test/outputs/ANN" # Changed output directory name
# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save each trained Keras model in .h5 format
try:
    model_d1.save(os.path.join(output_dir, "model_d1.h5"))
    model_d2.save(os.path.join(output_dir, "model_d2.h5"))
    model_d3.save(os.path.join(output_dir, "model_d3.h5"))
    model_total.save(os.path.join(output_dir, "model_total.h5")) # Save the total model
    joblib.dump(scaler, os.path.join(output_dir, "scaler_optimized.pkl")) # Save the scaler

    # KEEP this final print statement
    print(f"\nTất cả mô hình và scaler đã được lưu vào thư mục: {output_dir}")
except Exception as e:
    # KEEP this error print statement
    print(f"\nLỗi khi lưu mô hình hoặc scaler: {e}")
