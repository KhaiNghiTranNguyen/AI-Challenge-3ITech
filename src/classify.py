import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class FoodClassifier:
    def __init__(self, model_path=None, class_names=None):
        """
        Khởi tạo bộ phân loại thực phẩm với bất kỳ mô hình h5 nào
        
        Args:
            model_path (str, optional): Đường dẫn đến mô hình đã huấn luyện.
            class_names (list, optional): Danh sách tên các lớp thực phẩm. 
        """
        self.class_names = class_names
        
        # Tải mô hình từ đường dẫn được chỉ định hoặc tìm kiếm trong các vị trí mặc định
        try:
            if model_path and os.path.exists(model_path):
                print(f"Đang tải mô hình từ {model_path}")
                self.model = load_model(model_path, compile=False)
            else:
                # Tìm kiếm mô hình ở các vị trí mặc định
                default_paths = [
                    'models/cnn.h5',
                    'cnn.h5',
                ]
                
                for path in default_paths:
                    if os.path.exists(path):
                        print(f"Đang tải mô hình từ {path}")
                        self.model = load_model(path, compile=False)
                        break
                else:
                    raise FileNotFoundError("Không tìm thấy mô hình")
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {str(e)}")
            raise
    
    def preprocess_image(self, image_path):
        """
        Tiền xử lý ảnh để đưa vào mô hình
        
        Args:
            image_path (str): Đường dẫn đến ảnh cần phân loại
            
        Returns:
            numpy.ndarray: Ảnh đã qua tiền xử lý
        """
        try:
            # Xử lý đường dẫn ảnh
            if isinstance(image_path, str):
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Không thể đọc ảnh từ {image_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = image_path
                if img.shape[2] == 3 and img[0,0,0] > img[0,0,2]:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize và chuẩn bị ảnh
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            return img
        except Exception as e:
            print(f"Lỗi khi tiền xử lý ảnh: {str(e)}")
            raise
    
    def classify(self, image_path):
        """
        Phân loại ảnh thực phẩm
        
        Args:
            image_path (str or numpy.ndarray): Đường dẫn đến ảnh hoặc ảnh dạng numpy array
            
        Returns:
            str: Tên lớp dự đoán
        """
        try:
            # Tiền xử lý ảnh
            preprocessed_img = self.preprocess_image(image_path)
            
            # Dự đoán
            predictions = self.model.predict(preprocessed_img)
            predicted_class_index = np.argmax(predictions[0])
            
            # Lấy tên lớp
            if self.class_names and predicted_class_index < len(self.class_names):
                return self.class_names[predicted_class_index]
            else:
                return str(predicted_class_index)
        except Exception as e:
            print(f"Lỗi khi phân loại: {str(e)}")
            return None