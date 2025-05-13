import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

class ANNModel:
    def __init__(self):
        self.loadModel()

    def predict(self, toan_tb: float, van_tb: float, kh_ta_tb: float) -> list:
        """
        Dự đoán điểm dựa trên mô hình đã huấn luyện.

        Args:
            toan_tb (float): Điểm trung bình môn Toán.
            van_tb (float): Điểm trung bình môn Văn.
            kh_ta_tb (float): Điểm trung bình môn Khoa học hoặc Tiếng Anh.

        Returns:
            list[int]: Gồm 4 giá trị:
                - pred_d1 (int): Dự đoán điểm môn Toán.
                - pred_d2 (int): Dự đoán điểm môn Văn.
                - pred_d3 (int): Dự đoán điểm Khoa học hoặc Tiếng Anh.
                - predict_total (int): Tổng điểm dự đoán.
        """
        input_d1 = np.array([[toan_tb]])
        input_d2 = np.array([[van_tb]])
        input_d3 = np.array([[kh_ta_tb]])

        pred_d1 = self.model_d1.predict(input_d1)[0][0]
        pred_d2 = self.model_d2.predict(input_d2)[0][0]
        pred_d3 = self.model_d3.predict(input_d3)[0][0]
        
        pred_d1, pred_d2, pred_d3 = [max(round(val, 0), 0) for val in [pred_d1, pred_d2, pred_d3]]
        predict_total = pred_d1 + pred_d2 + pred_d3
        return [pred_d1, pred_d2, pred_d3, predict_total]

    def loadModel(self):
        base_path = os.path.abspath(os.path.dirname(__file__))  # Lấy đường dẫn thư mục hiện tại
        model_path_d1 = os.path.join(base_path, "model_d1.pkl")
        model_path_d2 = os.path.join(base_path, "model_d2.pkl")
        model_path_d3 = os.path.join(base_path, "model_d3.pkl")

        self.model_d1 = joblib.load(model_path_d1)
        self.model_d2 = joblib.load(model_path_d2)
        self.model_d3 = joblib.load(model_path_d3)

    def info(self) -> str:
        return "Artificial Neural Network Model - HSA"
