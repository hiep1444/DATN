import joblib
import pandas as pd

class SVMModel:
    def __init__(self):
        self.loadModel()

    def predict(self, toan_tb: float, van_tb: float, kh_ta_tb: float, type_mon3: int) -> list[int]:
        """
        Dự đoán điểm dựa trên mô hình đã huấn luyện.

        Args:
            toan_tb (float): Điểm trung bình môn Toán.
            van_tb (float): Điểm trung bình môn Văn.
            kh_ta_tb (float): Điểm trung bình môn Khoa học
            type_mon3 (int): Loại môn thứ 3 Khoa học

        Returns:
            list[int]: Gồm 4 giá trị:
                - pred_d1 (int): Dự đoán điểm môn Toán.
                - pred_d2 (int): Dự đoán điểm môn Văn.
                - pred_d3 (int): Dự đoán điểm Khoa học 
                - predict_total (int): Tổng điểm dự đoán.
        """
        input_d1 = pd.DataFrame([[toan_tb]], columns=['Toán_TB'])
        input_d2 = pd.DataFrame([[van_tb]], columns=['Văn_TB'])
        input_d3 = pd.DataFrame([[kh_ta_tb]], columns=['KH_TB'])

        pred_d1 = self.model_d1.predict(input_d1)[0]
        pred_d2 = self.model_d2.predict(input_d2)[0]
        pred_d3 = self.model_d3.predict(input_d3)[0]
        pred_d1, pred_d2, pred_d3 = [max(round(val, 0), 0) for val in [pred_d1, pred_d2, pred_d3]]
        predict_total = pred_d1 + pred_d2 + pred_d3
        return [pred_d1, pred_d2, pred_d3, predict_total]

    
    def loadModel(self):
        self.model_d1 = joblib.load("./outputs/SVM/model_d1.pkl")
        self.model_d2 = joblib.load("./outputs/SVM/model_d2.pkl")
        self.model_d3 = joblib.load("./outputs/SVM/model_d3.pkl")

    def info(self) -> str:
        return "SVM Model - HSA"

