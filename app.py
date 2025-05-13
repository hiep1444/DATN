import os
import uuid
import pandas as pd
from flask import Flask, jsonify, render_template, request
from outputs.LinearRegression.LinearRegressionModel import LinearRegressionModel
from outputs.SVM.SVMModel import SVMModel
from outputs.RandomForest.RandomForestModel import RandomForestModel
# from outputs.ANN.ANNModel import ANNModel
from models.ParseFile import ReaderCsv # Giả định ReaderCsv đã hoạt động đúng cho CSV

# Cho phép user chọn model để dự đoán
models = {
    "LinearRegression": LinearRegressionModel(),
    "SVM": SVMModel(),
    "RandomForest": RandomForestModel(),
    # "ANN": ANNModel(),
}

# Model tốt nhất nếu user chọn model mặc định là Tự động
model_HSA = models["SVM"]

currentTask = {}

def unique_id():
    return str(uuid.uuid4())

app = Flask(__name__)

@app.route('/')
def home():
    list_models = list(models.keys())
    return render_template('input.html', models = list_models)

@app.route('/predict')
def about():
    list_models = list(models.keys())
    return render_template('input.html', models = list_models)

@app.route('/predict', methods=['POST'])
def predict():
    global currentTask
    model_using = model_HSA
    try:
        toan_tb = float(request.form.get('ToanTB', 0))
        van_tb = float(request.form.get('VanTB', 0))
        kh_ta_tb = float(request.form.get('KH_TA_TB', 0))
        type_mon3 = int(request.form.get('TypeMon3', 1))
        model_type = request.form.get('ModelUsing', 'SVM')
        if model_type != "0":
            try:
                model_using = models[model_type]
            except KeyError:
                return render_template('message.html', message="Model không hợp lệ. Vui lòng kiểm tra lại!")

    except ValueError:
        return render_template('message.html', message="Dữ liệu nhập vào không hợp lệ. Vui lòng kiểm tra lại!")

    # Thêm kiểm tra phạm vi điểm cho nhập liệu trực tiếp
    if not (0 <= toan_tb <= 10 and 0 <= van_tb <= 10 and 0 <= kh_ta_tb <= 10):
         return render_template('message.html', message="Điểm trung bình môn không hợp lệ. Vui lòng kiểm tra lại!")


    print(f"Toán: {toan_tb}, Văn: {van_tb}, KH/TA: {kh_ta_tb}")

    result = model_using.predict(toan_tb, van_tb, kh_ta_tb, type_mon3)
    uid = unique_id()
    currentTask[uid] = {
        "toan_tb": toan_tb,
        "van_tb": van_tb,
        "kh_ta_tb": kh_ta_tb,
        "type_mon3": type_mon3,
        "result": result,
        "model": model_using.info()
    }

    return render_template('result.html',
                           toan_tb=toan_tb,
                           van_tb=van_tb,
                           kh_ta_tb=kh_ta_tb,
                           pred_d1=result[0],
                           pred_d2=result[1],
                           pred_d3=result[2],
                           predict_total=result[3],
                           task_id=uid,
                           model=model_using.info())


@app.route('/review', methods=['POST'])
def review():
    task_id = request.form.get('task_id')
    review = request.form.get('Review')
    comment = request.form.get('Comment')

    if task_id not in currentTask:
        return "Task ID không hợp lệ", 400

    task_data = currentTask[task_id]
    toan_tb = task_data["toan_tb"]
    van_tb = task_data["van_tb"]
    kh_ta_tb = task_data["kh_ta_tb"]
    pred_d1 = task_data["result"][0]
    pred_d2 = task_data["result"][1]
    pred_d3 = task_data["result"][2]
    predict_total = task_data["result"][3]
    model_name = task_data["model"]

    review_data = pd.DataFrame([{
        "Model": model_name,
        "Toán TB User": toan_tb,
        "Văn TB User": van_tb,
        "D3 User": kh_ta_tb,
        "Toán Predict": pred_d1,
        "Văn Predict": pred_d2,
        "D3 Predict": pred_d3,
        "Đúng/Sai": review,
        "Nhận xét": comment,
        "Timestamp": pd.Timestamp.now()
    }])

    output_file = "./outputs/result.csv"

    # Sử dụng 'a' mode để append, header=False cho các dòng sau dòng đầu tiên
    if not os.path.exists(output_file):
        review_data.to_csv(output_file, index=False, encoding='utf-8-sig')
    else:
        review_data.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
    print(f"Task ID: {task_id}, Review: {review}, Comment: {comment}")
    del currentTask[task_id]
    return render_template("message.html", message="Cảm ơn bạn đã sử dụng hệ thống dự đoán điểm HSA của chúng tôi. Chúc bạn một ngày tốt lành!")


@app.route('/upload', methods=['POST'])
def upload_file():
    model_type = request.form.get('ModelUsing', 'SVM')
    model_using = model_HSA # default

    try:
        if model_type != "0":
            model_using = models[model_type]
    except KeyError:
        return render_template('message.html', message="Model không hợp lệ. Vui lòng kiểm tra lại!")

    file = request.files['file']
    if not file:
        return render_template('message.html', message="Không có file nào được tải lên.")

    filename = file.filename
    file_extension = filename.split('.')[-1].lower()

    result = []
    error_message = None
    error_count = 0
    total_rows_processed = 0

    if file_extension == 'csv':
        # Sử dụng ReaderCsv hiện có để xử lý CSV
        reader = ReaderCsv(file)
        error_message = reader.error_message
        error_count = reader.error_rows

        if not error_message and len(reader.data) > 0:
             total_rows_processed = len(reader.data)
             for data in reader.data:
                try:
                    # Giả định ReaderCsv trả về list các giá trị
                    toan_tb = float(data[0])
                    van_tb = float(data[1])
                    kh_ta_tb = float(data[2])
                    # Lấy type_mon3 nếu có, nếu không mặc định là 1
                    type_mon3 = int(data[3]) if len(data) > 3 and data[3] != '' else 1

                    # Kiểm tra phạm vi điểm
                    if not (0 <= toan_tb <= 10 and 0 <= van_tb <= 10 and 0 <= kh_ta_tb <= 10):
                         raise ValueError("Điểm nằm ngoài phạm vi cho phép (0-10).")

                    pred = model_using.predict(toan_tb, van_tb, kh_ta_tb, type_mon3)
                    result.append([toan_tb, van_tb, kh_ta_tb, pred[0], pred[1], pred[2], pred[3]])
                except (ValueError, IndexError, TypeError) as e:
                    error_count += 1
                    # print(f"Lỗi xử lý dòng CSV: {e}") # Debugging

    elif file_extension == 'xlsx':
        try:
            # Đọc file Excel: bỏ qua dòng đầu tiên (header), chỉ lấy cột A, B, C
            # header=None và skiprows=[0] đảm bảo dòng đầu tiên bị bỏ qua và không dùng làm header
            # usecols='A:C' chỉ đọc 3 cột đầu tiên
            df = pd.read_excel(file, header=None, skiprows=[0], usecols='A:C', engine='openpyxl')

            if df.empty:
                 error_message = "File Excel không có dữ liệu hợp lệ sau khi bỏ qua dòng đầu tiên."
            else:
                total_rows_processed = len(df)
                # Lặp qua từng dòng dữ liệu
                for index, row_data in df.iterrows():
                    try:
                        # Lấy giá trị từ các cột A, B, C (tương ứng index 0, 1, 2 sau khi đọc)
                        # .iloc[index] an toàn khi truy cập theo vị trí
                        toan_tb = float(row_data.iloc[0])
                        van_tb = float(row_data.iloc[1])
                        kh_ta_tb = float(row_data.iloc[2])
                        # type_mon3 không có trong 3 cột được yêu cầu, mặc định là 1
                        type_mon3 = 1

                        # Kiểm tra phạm vi điểm
                        if not (0 <= toan_tb <= 10 and 0 <= van_tb <= 10 and 0 <= kh_ta_tb <= 10):
                            raise ValueError("Điểm nằm ngoài phạm vi cho phép (0-10).")

                        pred = model_using.predict(toan_tb, van_tb, kh_ta_tb, type_mon3)
                        result.append([toan_tb, van_tb, kh_ta_tb, pred[0], pred[1], pred[2], pred[3]])
                    except (ValueError, IndexError, TypeError) as e:
                        error_count += 1
                        # print(f"Lỗi xử lý dòng {index + 2} (Excel): {e}") # Debugging: index + 2 để khớp với số dòng trong Excel (bỏ qua dòng 1, index từ 0)

        except Exception as e: # Bắt các lỗi có thể xảy ra khi đọc file Excel
             error_message = f"Lỗi khi đọc hoặc xử lý file Excel: {e}"
             # print(f"Lỗi khi đọc hoặc xử lý file Excel: {e}") # Debugging

    else:
        error_message = "Định dạng file không được hỗ trợ. Vui lòng tải lên file CSV hoặc XLSX."

    # Xử lý kết quả cuối cùng và hiển thị thông báo phù hợp
    if error_message:
         # Có lỗi nghiêm trọng khi đọc file
         return render_template("message.html", message=error_message)
    elif total_rows_processed == 0 and not result:
         # Không có dòng dữ liệu nào được xử lý thành công (có thể file rỗng, chỉ có header, hoặc tất cả các dòng đều lỗi)
         msg = "Không có dữ liệu hợp lệ để xử lý trong file."
         if error_count > 0:
             msg += f" Có {error_count} dòng bị lỗi."
         return render_template("message.html", message=msg)
    elif error_count > 0:
         # Có dữ liệu được xử lý thành công, nhưng có dòng bị lỗi
         warning_message = f"Hoàn thành xử lý dữ liệu. Có {error_count} dòng dữ liệu bị lỗi và không được xử lý."
         return render_template("result_table.html", result=result, model=model_using.info(), warning_message=warning_message)
    else:
        # Tất cả các dòng đều được xử lý thành công
        return render_template("result_table.html", result=result, model=model_using.info())


@app.route('/report_error', methods=['POST'])
def report_error():
    data = request.json

    toan_tb = data["toan_tb"]
    van_tb = data["van_tb"]
    kh_ta_tb = data["kh_ta_tb"]
    pred_d1 = data["pred_d1"]
    pred_d2 = data["pred_d2"]
    pred_d3 = data["pred_d3"]
    predict_total = data["predict_total"]
    model_name = data["model"]

    review_data = pd.DataFrame([{
        "Model": model_name,
        "Toán TB User": toan_tb,
        "Văn TB User": van_tb,
        "D3 User": kh_ta_tb,
        "Toán Predict": pred_d1,
        "Văn Predict": pred_d2,
        "D3 Predict": pred_d3,
        "Đúng/Sai": 0,
        "Nhận xét": "Báo lỗi",
        "Timestamp": pd.Timestamp.now()
    }])

    output_file = "./outputs/result.csv"

    if not os.path.exists(output_file):
        review_data.to_csv(output_file, index=False, encoding='utf-8-sig')
    else:
        review_data.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')

    return jsonify({"message": "Báo sai thành công!"})


if __name__ == '__main__':
    app.run(debug=True)