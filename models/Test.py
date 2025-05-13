import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Đọc dữ liệu từ file CSV
df = pd.read_csv("../data/HSA_HD_Final_Cleaned.csv")

# Tách feature và label
X = df[['Toán_TB', 'Văn_TB', 'KH_TB']]
y_d1 = df['d1']  # Tư duy định lượng (Toán học)
y_d2 = df['d2']  # Tư duy định tính (Văn học)
y_d3 = df['d3']  # Khoa học tự nhiên

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia tập train/test
X_train_d1, X_test_d1, y_d1_train, y_d1_test = train_test_split(X_scaled[:, [0]], y_d1, test_size=0.2, random_state=42)
X_train_d2, X_test_d2, y_d2_train, y_d2_test = train_test_split(X_scaled[:, [1]], y_d2, test_size=0.2, random_state=42)
X_train_d3, X_test_d3, y_d3_train, y_d3_test = train_test_split(X_scaled[:, [2]], y_d3, test_size=0.2, random_state=42)

# Tối ưu tham số cho SVR
param_grid = {'C': [0.1, 1, 10], 'epsilon': [0.1, 0.2, 0.5], 'kernel': ['linear', 'rbf']}

svr_d1 = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_absolute_error')
svr_d1.fit(X_train_d1, y_d1_train)
svr_d2 = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_absolute_error')
svr_d2.fit(X_train_d2, y_d2_train)
svr_d3 = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_absolute_error')
svr_d3.fit(X_train_d3, y_d3_train)

# Mô hình RandomForest để so sánh
rf_d1 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_d1.fit(X_train_d1, y_d1_train)
rf_d2 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_d2.fit(X_train_d2, y_d2_train)
rf_d3 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_d3.fit(X_train_d3, y_d3_train)

# Hàm đánh giá
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Dự đoán với SVR
print("SVR Results:")
evaluate_model(y_d1_test, svr_d1.best_estimator_.predict(X_test_d1), "SVR d1")
evaluate_model(y_d2_test, svr_d2.best_estimator_.predict(X_test_d2), "SVR d2")
evaluate_model(y_d3_test, svr_d3.best_estimator_.predict(X_test_d3), "SVR d3")

# Dự đoán với RandomForest
print("RandomForest Results:")
evaluate_model(y_d1_test, rf_d1.predict(X_test_d1), "RF d1")
evaluate_model(y_d2_test, rf_d2.predict(X_test_d2), "RF d2")
evaluate_model(y_d3_test, rf_d3.predict(X_test_d3), "RF d3")

# Kiểm tra độ chính xác trên tập dữ liệu mẫu
count = int(len(df) * 0.2)
test_data = df.sample(count)
cnt = 0
for _, row in test_data.iterrows():
    input_scaled = scaler.transform([[row['Toán_TB'], row['Văn_TB'], row['KH_TB']]])
    pred_d1 = svr_d1.best_estimator_.predict([[input_scaled[0][0]]])[0]
    pred_d2 = svr_d2.best_estimator_.predict([[input_scaled[0][1]]])[0]
    pred_d3 = svr_d3.best_estimator_.predict([[input_scaled[0][2]]])[0]
    predict_total = pred_d1 + pred_d2 + pred_d3
    expected_total = row['diem']
    cnt += abs(predict_total - expected_total) <= 15

print(f"Độ chính xác (SVR): {cnt/count*100:.2f}%")