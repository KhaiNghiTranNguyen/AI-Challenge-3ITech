import os
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

class FoodDetector:
    def __init__(self, model_path=None):
        """
        Khởi tạo bộ phát hiện thực phẩm sử dụng YOLO
        
        Args:
            model_path (str, optional): Đường dẫn đến trọng số YOLO tùy chỉnh. Nếu None, sử dụng model mặc định.
        """
        # Nếu không chỉ định path cụ thể, sử dụng mô hình YOLOv8n mặc định
        model_to_load = model_path if model_path else 'yolov8n.pt'
        print(f"Đang tải mô hình YOLO từ: {model_to_load}")
        self.model = YOLO(model_to_load)

    def detect_and_crop(self, image_path):
        """
        Phát hiện bowl/đĩa thức ăn trong ảnh và cắt ra từng phần riêng biệt
        
        Args:
            image_path (str): Đường dẫn đến ảnh cần phân tích
            
        Returns:
            tuple: (danh sách đường dẫn ảnh đã cắt, danh sách tên lớp, kết quả YOLO)
        """
        print(f"Đang phát hiện món ăn trong ảnh: {image_path}")
        
        # Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
        
        # Phát hiện đối tượng với YOLOv8
        results = self.model(img)
        result = results[0]  # Lấy kết quả của ảnh đầu tiên
        
        # Lấy tọa độ và lớp của các đối tượng đã phát hiện
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        classes = result.boxes.cls.cpu().numpy()  # ID lớp
        conf_scores = result.boxes.conf.cpu().numpy()  # Độ tin cậy
        
        print(f"Đã phát hiện {len(boxes)} đối tượng")
        
        # Lọc các đối tượng là bowl (class 45) hoặc plate hoặc các món ăn
        # Lưu ý: YOLO có thể phát hiện nhiều class khác nhau, chúng ta tập trung vào bowl (45) 
        # hoặc các đối tượng liên quan đến thức ăn
        cropped_paths = []
        yolo_classes = []
        filtered_boxes = []
        
        # Tạo thư mục tạm thời để lưu các ảnh đã cắt
        temp_dir = tempfile.mkdtemp()
        
        for i, (box, cls, conf) in enumerate(zip(boxes, classes, conf_scores)):
            # Lọc đối tượng là bowl (class 45) hoặc đồ ăn
            # Nếu confidence score thấp, bỏ qua
            if conf < 0.4:
                continue
                
            # Ưu tiên phát hiện bowl (class 45) và các đối tượng liên quan đến thức ăn
            if cls == 45:  # class 45 là bowl trong COCO dataset
                x1, y1, x2, y2 = map(int, box)
                
                # Đảm bảo tọa độ nằm trong giới hạn ảnh
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                
                # Cắt ảnh
                cropped_img = img[y1:y2, x1:x2]
                
                # Đảm bảo ảnh cắt không rỗng
                if cropped_img.size == 0 or cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
                    print(f"  Bỏ qua bounding box {i} vì ảnh cắt rỗng")
                    continue
                
                # Lưu ảnh đã cắt
                crop_path = os.path.join(temp_dir, f"crop_{i}.jpg")
                cv2.imwrite(crop_path, cropped_img)
                
                cropped_paths.append(crop_path)
                yolo_classes.append("bowl")  # Gán nhãn tạm thời
                filtered_boxes.append(box)
                
                print(f"  Đã lưu ảnh cắt {i}: {crop_path}, kích thước: {cropped_img.shape}")
        
        print(f"Tổng số món ăn đã phát hiện và cắt: {len(cropped_paths)}")
        
        return cropped_paths, yolo_classes, results

if __name__ == "__main__":
    # Demo
    detector = FoodDetector()
    image_path = "đường_dẫn_đến_ảnh_thử_nghiệm.jpg"
    try:
        cropped_paths, classes, results = detector.detect_and_crop(image_path)
        print(f"Đã phát hiện {len(cropped_paths)} món ăn:")
        for i, (path, cls) in enumerate(zip(cropped_paths, classes)):
            print(f"  {i+1}. {cls}: {path}")
    except Exception as e:
        print(f"Lỗi: {str(e)}")