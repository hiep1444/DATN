# Dữ kiện
* Tuân theo chuẩn bài thi đánh giá năng lực HSA của Đại học quốc gia Hà Nội (VNU - CET). Trong đó:
Điểm d1 là điểm Tư duy định lượng - Lĩnh vực Toán học
Điểm d2 là điểm Tư duy định tính - Lĩnh vực Văn học
Điểm d3 là điểm Khoa học hoặc Tiếng Anh - Lĩnh vực Khoa học tự nhiên, Xã hội hoặc Tiếng Anh


* Input mong muốn: Nhập trực tiếp điểm trên web và up File template (cả 2)
* Output mong muốn: Kết quả d1,d2,d3, điểm tổng.
* Các mô hình cần so sánh
# Công việc cần làm
* Xác nhận dữ liệu: Đâu là điểm Tư duy định lượng, định tính, Khoa học hoặc Tiếng Anh trong các điểm d1, d2, d3
* Giao diện web (Mockup giao diện)
* Mô hình dự đoán
* So sánh các mô hình (SVM, RandomForest, Decision Tree)
* Có 1 số giả thiết cần xác nhận: Kiến thức bài thi ĐGNL HSA có nhiều kiến thức năm lớp 11, 12 hơn, cần xác minh thông tin này (dẫn chứng bài báo cụ thể) ??

# Tiến trình dự án
## Tiền xử lý dữ liệu
* File dữ liệu có: Mã ĐTN, Giới tính, Học lực, Hạnh kiểm, Điểm từng môn lớp 10 - 11 - 12, Điểm HSA (d1, d2, d3)
* Xóa cột không có giá trị sử dụng: Mã ĐTN, Giới tính, Học lực (xét HSA trên từng môn riêng lẻ nên học lực không quan trọng), Hạnh kiểm, Điểm từng kì (Chỉ cần xét điểm từng môn trên cả năm là được)
* Tính giá trị điểm trung bình cả 3 năm dựa trên công thức: TB = (Điểm CN 10 + Điểm CN 11 + Điểm CN 12 * 2)/4 (Ưu tiên điểm lớp 12 - hệ số 2)
Trong đó CN = Cả năm
* Tính giá trị điểm Khoa học xã hội, Khoa học tự nhiên dựa trên trung bình cộng tổ hợp của các môn tương ứng
* Kết quả xem ở file data-preprocessing.ipynb
