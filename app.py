from flask import Flask, render_template, request, jsonify
import os
import base64
import tempfile
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import json

# Import các module hiện có
from src.detect import FoodDetector
from src.classify import FoodClassifier
from src.billing import BillingSystem

app = Flask(__name__, 
            static_folder='web_ui/static',
            template_folder='web_ui/templates')
            
# Tăng giới hạn kích thước nội dung lên 50MB
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Cấu hình tuần tự hóa JSON để xử lý các kiểu dữ liệu NumPy
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(CustomJSONEncoder, self).default(obj)

app.json_encoder = CustomJSONEncoder

# Khởi tạo hệ thống
class_names = [
    'banh mi', 'bap cai luoc', 'bap cai xao', 'bo xao', 'ca chien', 'ca chua', 'ca kho',
    'ca rot', 'canh bau', 'canh bi do', 'canh cai', 'canh chua', 'canh rong bien', 'chuoi',
    'com', 'dau bap', 'dau hu', 'dau que', 'do chua', 'dua hau', 'dua leo', 'ga chien',
    'ga kho', 'kho qua', 'kho tieu', 'kho trung', 'nuoc mam', 'nuoc tuong', 'oi', 'ot',
    'rau', 'rau muong', 'rau ngo', 'suon mieng', 'suon xao', 'thanh long', 'thit chien',
    'thit luoc', 'tom', 'trung chien', 'trung luoc'
]

detector = None
classifier = None
billing = None

def initialize_models():
    global detector, classifier, billing
    detector = FoodDetector()
    classifier = FoodClassifier(class_names=class_names)
    billing = BillingSystem()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    # Kiểm tra xem các mô hình đã được khởi tạo chưa
    if detector is None or classifier is None or billing is None:
        initialize_models()
    
    try:
        # Lấy dữ liệu ảnh từ request
        if 'image' not in request.files:
            # Xử lý ảnh được mã hóa base64
            if 'imageData' in request.form:
                image_data = request.form['imageData']
                # Loại bỏ tiền tố data URL nếu có
                if 'data:image' in image_data:
                    image_data = image_data.split(',')[1]
                
                try:
                    # Chuyển đổi base64 thành ảnh
                    image_bytes = base64.b64decode(image_data)
                    
                    # Tạo file tạm để lưu ảnh
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    temp_file.write(image_bytes)
                    temp_file.close()
                    image_path = temp_file.name
                except Exception as e:
                    return jsonify({'error': f'Lỗi xử lý dữ liệu ảnh: {str(e)}'}), 400
            else:
                return jsonify({'error': 'Không có ảnh được cung cấp'}), 400
        else:
            # Xử lý tải lên file
            image_file = request.files['image']
            if not image_file:
                return jsonify({'error': 'File rỗng'}), 400
                
            try:
                filename = secure_filename(image_file.filename)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
                image_file.save(temp_file.name)
                temp_file.close()
                image_path = temp_file.name
            except Exception as e:
                return jsonify({'error': f'Lỗi lưu file đã tải lên: {str(e)}'}), 400
        
        # Xác thực xem ảnh có thể đọc được không
        try:
            img = cv2.imread(image_path)
            if img is None:
                return jsonify({'error': 'Không thể đọc file ảnh. Vui lòng thử ảnh khác.'}), 400
                
            # Resize ảnh nếu quá lớn
            max_dimension = 1920  # Kích thước tối đa cho xử lý
            height, width = img.shape[:2]
            print(f"Kích thước ảnh gốc: {width}x{height}")
            if height > max_dimension or width > max_dimension:
                if height > width:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                else:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                img = cv2.resize(img, (new_width, new_height))
                cv2.imwrite(image_path, img)
                print(f"Đã resize ảnh thành {new_width}x{new_height}")
        except Exception as e:
            return jsonify({'error': f'Lỗi đọc ảnh: {str(e)}'}), 400
        
        # Xử lý ảnh sử dụng pipeline hiện có
        try:
            print("Bắt đầu phát hiện thực phẩm với YOLO...")
            # Phát hiện các đối tượng bowl (class 45) và cắt
            cropped_paths, yolo_classes, results = detector.detect_and_crop(image_path)
            print(f"Hoàn tất phát hiện YOLO. Tìm thấy {len(cropped_paths)} món.")
            for i, cls in enumerate(yolo_classes):
                print(f"  Món {i+1}: {cls}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': f'Lỗi phát hiện món ăn: {str(e)}'
            }), 500
        
        # Kiểm tra xem có món ăn nào được phát hiện không
        if len(cropped_paths) == 0:
            return jsonify({
                'error': 'Không phát hiện được món ăn nào trong hình ảnh. Vui lòng thử lại với ảnh khác.',
                'items_found': 0
            }), 400
        
        # Phân loại từng ảnh đã cắt và tạo danh sách food_items
        print("Bắt đầu phân loại các ảnh đã cắt...")
        food_items = []
        detected_items = []
        
        for i, (path, yolo_class) in enumerate(zip(cropped_paths, yolo_classes)):
            try:
                print(f"Xử lý món {i+1}: {yolo_class} (lớp YOLO)")
                # Tải ảnh đã cắt để hiển thị
                crop_img = cv2.imread(path)
                if crop_img is None:
                    print(f"  Không thể đọc ảnh đã cắt tại {path}")
                    continue
                
                # Nén và resize ảnh đã cắt để giảm kích thước
                max_crop_size = 300
                height, width = crop_img.shape[:2]
                if height > max_crop_size or width > max_crop_size:
                    if height > width:
                        new_height = max_crop_size
                        new_width = int(width * (max_crop_size / height))
                    else:
                        new_width = max_crop_size
                        new_height = int(height * (max_crop_size / width))
                    crop_img = cv2.resize(crop_img, (new_width, new_height))
                
                # Mã hóa ảnh thành dữ liệu base64
                _, buffer = cv2.imencode('.jpg', crop_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                crop_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Phân loại với CNN
                print(f"  Phân loại với CNN...")
                cnn_class = classifier.classify(path)
                print(f"  Kết quả phân loại CNN: {cnn_class}")
                
                if cnn_class:
                    food_class = cnn_class
                    detected_items.append({
                        'id': i,
                        'yolo_class': yolo_class,
                        'final_class': cnn_class,
                        'image': f'data:image/jpeg;base64,{crop_b64}'
                    })
                    food_items.append(cnn_class)
                else:
                    food_class = yolo_class
                    detected_items.append({
                        'id': i,
                        'yolo_class': yolo_class,
                        'final_class': yolo_class,
                        'image': f'data:image/jpeg;base64,{crop_b64}'
                    })
                    food_items.append(yolo_class)
            except Exception as e:
                print(f"Lỗi xử lý ảnh đã cắt {i}: {e}")
                continue
        
        # Nếu không xử lý được món ăn nào
        if len(food_items) == 0:
            return jsonify({
                'error': 'Không thể xử lý được món ăn nào. Vui lòng thử lại với ảnh rõ ràng hơn.',
                'items_processed': 0
            }), 400
        
        try:
            # Tính hóa đơn
            print("Tính hóa đơn cho các món:", food_items)
            bill_details, total_cost, total_calories = billing.calculate_bill(food_items)
            print(f"Hoàn tất tính hóa đơn: {len(bill_details)} món, {total_cost} VND, {total_calories} kcal")
            
            # Chuyển đổi kiểu dữ liệu NumPy thành kiểu Python
            total_cost = float(total_cost) if hasattr(total_cost, 'item') else total_cost
            total_calories = float(total_calories) if hasattr(total_calories, 'item') else total_calories
            
            # Định dạng chi tiết hóa đơn để hiển thị tốt hơn
            formatted_bill = []
            for i, detail in enumerate(bill_details):
                price = float(detail['price']) if hasattr(detail['price'], 'item') else detail['price']
                calories = float(detail['calories']) if hasattr(detail['calories'], 'item') else detail['calories']
                
                print(f"Món {i+1}: {detail['item']}, Giá: {price} VND, Calo: {calories} kcal")
                
                item_details = {
                    'id': i,
                    'item': detail['item'],
                    'price': price,
                    'calories': calories,
                    'image': detected_items[i]['image'] if i < len(detected_items) else None
                }
                formatted_bill.append(item_details)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Lỗi tính hóa đơn: {str(e)}'}), 500
        
        # Dọn dẹp các file tạm
        try:
            os.unlink(image_path)
            for path in cropped_paths:
                if os.path.exists(path):
                    os.unlink(path)
        except Exception as e:
            print(f"Lỗi dọn dẹp file tạm: {e}")
        
        # Trả về kết quả
        result = {
            'success': True,
            'detected_items': detected_items,
            'bill_details': formatted_bill,
            'total_cost': total_cost,
            'total_calories': total_calories,
            'items_count': len(formatted_bill)
        }
        print("Kết quả cuối cùng:", result)
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Lỗi phân tích: {str(e)}'}), 500

@app.route('/api/food-info', methods=['GET'])
def get_food_info():
    """Trả về thông tin về các món ăn có sẵn"""
    try:
        if billing is None:
            initialize_models()
        
        # Lấy thông tin món ăn từ CSV menu
        menu_data = billing.menu.to_dict('records')
        
        # Định dạng dữ liệu cho frontend
        food_info = []
        for item in menu_data:
            food_info.append({
                'name': item['item'],
                'calories': item['calories'],
                'price': item['price'],
                'category': get_food_category(item['item'])
            })
        
        return jsonify({'success': True, 'food_info': food_info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update-food-item', methods=['POST'])
def update_food_item():
    """Cập nhật phân loại món ăn và tính lại hóa đơn"""
    try:
        # Lấy dữ liệu request
        data = request.json
        if not data or 'itemIndex' not in data or 'newFoodItem' not in data:
            return jsonify({'error': 'Thiếu dữ liệu cần thiết'}), 400
            
        item_index = data['itemIndex']
        new_food_item = data['newFoodItem']
        bill_data = data.get('billData', {})
        
        # Nếu chưa khởi tạo hệ thống tính tiền, khởi tạo nó
        if billing is None:
            initialize_models()
        
        # Lấy dữ liệu menu để tìm thông tin món ăn mới
        menu_data = billing.menu.to_dict('records')
        matching_item = next((item for item in menu_data if item['item'] == new_food_item), None)
        
        if not matching_item:
            return jsonify({'error': f'Không tìm thấy món {new_food_item} trong menu'}), 404
            
        # Tính lại hóa đơn với món ăn đã cập nhật
        # Nếu bill_data được cung cấp, sử dụng nó để cập nhật món cụ thể
        if bill_data and 'bill_details' in bill_data and item_index < len(bill_data['bill_details']):
            # Lưu thông tin cũ để tham khảo
            old_item = bill_data['bill_details'][item_index]['item']
            old_price = bill_data['bill_details'][item_index]['price']
            old_calories = bill_data['bill_details'][item_index]['calories']
            
            # Cập nhật món ăn
            bill_data['bill_details'][item_index]['item'] = new_food_item
            bill_data['bill_details'][item_index]['price'] = matching_item['price']
            bill_data['bill_details'][item_index]['calories'] = matching_item['calories']
            
            # Tính lại tổng
            total_cost = sum(item['price'] for item in bill_data['bill_details'])
            total_calories = sum(item['calories'] for item in bill_data['bill_details'])
            
            # Cập nhật tổng
            bill_data['total_cost'] = total_cost
            bill_data['total_calories'] = total_calories
            
            return jsonify({
                'success': True, 
                'updated_bill': bill_data,
                'old_item': {
                    'name': old_item,
                    'price': old_price,
                    'calories': old_calories
                },
                'new_item': {
                    'name': new_food_item,
                    'price': matching_item['price'],
                    'calories': matching_item['calories']
                }
            })
        else:
            # Trả về chỉ thông tin món ăn mới nếu không cung cấp bill_data
            return jsonify({
                'success': True,
                'new_item': {
                    'name': new_food_item,
                    'price': matching_item['price'],
                    'calories': matching_item['calories']
                }
            })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Lỗi cập nhật món ăn: {str(e)}'}), 500

def get_food_category(item_name):
    """Hàm đơn giản để phân loại các món ăn"""
    item_name = item_name.lower()
    
    # Phân loại dựa trên các loại thực phẩm Việt Nam phổ biến
    if any(x in item_name for x in ['com', 'bun', 'pho', 'banh']):
        return 'Carbohydrates'
    elif any(x in item_name for x in ['thit', 'ga', 'bo', 'ca chien', 'ca kho', 'trung', 'tom', 'suon', 'kho tieu', 'dau hu']):
        return 'Protein'
    elif any(x in item_name for x in ['rau', 'dau', 'bap cai', 'cai', 'dua leo', 'ot', 'ca chua', 'kho qua']):
        return 'Vegetable'
    elif any(x in item_name for x in ['canh', 'ca chua', 'ca rot']):
        return 'Soup'
    else:
        return 'Other'

if __name__ == '__main__':
    # Khởi tạo mô hình khi khởi động
    initialize_models()
    # Chạy ứng dụng Flask
    app.run(debug=True, host='0.0.0.0', port=5000)