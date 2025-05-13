import csv
from flask import request

class ReaderCsv:
    def __init__(self, file):
        self.file = file
        self.data = []
        self.error_rows = 0
        self.error_message = ""
        self.read_csv()

    def read_csv(self):
        try:
            stream = self.file.stream.read().decode("utf-8").splitlines()
            reader = csv.reader(stream)
            
            # Thử đọc dòng đầu tiên, nếu không parse được thì bỏ qua
            first_row = next(reader, None)
            try:
                _ = [float(value) for value in first_row]
                self.data.append(_)
            except ValueError:
                pass  # Dòng đầu không hợp lệ (là tiêu đề, không thể parse sang float) thì bỏ qua
            
            for row in reader:
                try:
                    self.data.append([float(value) for value in row])
                except ValueError:
                    self.error_rows += 1
        except Exception as e:
            self.error_message = f"Lỗi đọc file CSV: {e}"
            return
        

