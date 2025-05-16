import pandas as pd
import numpy as np
import os

class BillingSystem:
    def __init__(self, menu_path=None):
        """
        Khởi tạo hệ thống tính tiền dựa trên CSV menu
        
        Args:
            menu_path (str, optional): Đường dẫn đến file CSV menu. Nếu None, sử dụng vị trí mặc định.
        """
        # Nếu không chỉ định path cụ thể, tìm kiếm menu_info.csv ở các vị trí thông dụng
        if menu_path is None:
            potential_paths = [
                'menu_info.csv',
                os.path.join(os.path.dirname(__file__), 'menu_info.csv'),
                os.path.join(os.path.dirname(__file__), '..', 'menu_info.csv'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'menu_info.csv'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'menu_info.csv')
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    menu_path = path
                    break
            
            if menu_path is None:
                # Nếu không tìm thấy file, tạo một menu mặc định
                print("CẢNH BÁO: Không tìm thấy file menu_info.csv. Sử dụng menu mặc định.")
                self.menu = self._create_default_menu()
            else:
                print(f"Đang tải menu từ: {menu_path}")
                self.menu = pd.read_csv(menu_path)
        else:
            # Sử dụng đường dẫn đã cung cấp
            if os.path.exists(menu_path):
                self.menu = pd.read_csv(menu_path)
            else:
                print(f"CẢNH BÁO: Không tìm thấy file menu tại {menu_path}. Sử dụng menu mặc định.")
                self.menu = self._create_default_menu()
        
        # Đảm bảo các thông tin cần thiết có trong menu
        required_columns = ['item', 'price', 'calories']
        if not all(col in self.menu.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in self.menu.columns]
            print(f"CẢNH BÁO: Thiếu các cột trong menu: {missing_cols}. Sử dụng menu mặc định.")
            self.menu = self._create_default_menu()
    
    def _create_default_menu(self):
        """
        Tạo menu mặc định với giá và calo cho các món ăn phổ biến
        
        Returns:
            pandas.DataFrame: Menu mặc định
        """
        # Danh sách món ăn phổ biến với giá và calo tương ứng
        default_items = [
            {'item': 'banh mi', 'price': 15000, 'calories': 350},
            {'item': 'bap cai luoc', 'price': 5000, 'calories': 30},
            {'item': 'bap cai xao', 'price': 8000, 'calories': 60},
            {'item': 'bo xao', 'price': 25000, 'calories': 250},
            {'item': 'ca chien', 'price': 20000, 'calories': 200},
            {'item': 'ca chua', 'price': 5000, 'calories': 20},
            {'item': 'ca kho', 'price': 18000, 'calories': 180},
            {'item': 'ca rot', 'price': 5000, 'calories': 25},
            {'item': 'canh bau', 'price': 10000, 'calories': 40},
            {'item': 'canh bi do', 'price': 10000, 'calories': 45},
            {'item': 'canh cai', 'price': 10000, 'calories': 35},
            {'item': 'canh chua', 'price': 12000, 'calories': 60},
            {'item': 'canh rong bien', 'price': 12000, 'calories': 30},
            {'item': 'chuoi', 'price': 5000, 'calories': 90},
            {'item': 'com', 'price': 10000, 'calories': 150},
            {'item': 'dau bap', 'price': 7000, 'calories': 35},
            {'item': 'dau hu', 'price': 8000, 'calories': 70},
            {'item': 'dau que', 'price': 7000, 'calories': 30},
            {'item': 'do chua', 'price': 5000, 'calories': 15},
            {'item': 'dua hau', 'price': 8000, 'calories': 50},
            {'item': 'dua leo', 'price': 5000, 'calories': 15},
            {'item': 'ga chien', 'price': 22000, 'calories': 280},
            {'item': 'ga kho', 'price': 20000, 'calories': 250},
            {'item': 'kho qua', 'price': 8000, 'calories': 25},
            {'item': 'kho tieu', 'price': 18000, 'calories': 200},
            {'item': 'kho trung', 'price': 12000, 'calories': 150},
            {'item': 'nuoc mam', 'price': 2000, 'calories': 10},
            {'item': 'nuoc tuong', 'price': 2000, 'calories': 5},
            {'item': 'oi', 'price': 8000, 'calories': 40},
            {'item': 'ot', 'price': 2000, 'calories': 5},
            {'item': 'rau', 'price': 5000, 'calories': 20},
            {'item': 'rau muong', 'price': 8000, 'calories': 25},
            {'item': 'rau ngo', 'price': 5000, 'calories': 15},
            {'item': 'suon mieng', 'price': 25000, 'calories': 300},
            {'item': 'suon xao', 'price': 25000, 'calories': 280},
            {'item': 'thanh long', 'price': 10000, 'calories': 60},
            {'item': 'thit chien', 'price': 22000, 'calories': 250},
            {'item': 'thit luoc', 'price': 20000, 'calories': 180},
            {'item': 'tom', 'price': 25000, 'calories': 120},
            {'item': 'trung chien', 'price': 10000, 'calories': 120},
            {'item': 'trung luoc', 'price': 8000, 'calories': 80},
            # Thêm một món ăn "không xác định" để xử lý các trường hợp không nhận dạng được
            {'item': 'unknown', 'price': 10000, 'calories': 100}
        ]
        
        return pd.DataFrame(default_items)
    
    def calculate_bill(self, food_items):
        """
        Tính toán hóa đơn dựa trên danh sách các món ăn
        
        Args:
            food_items (list): Danh sách các món ăn đã phát hiện
            
        Returns:
            tuple: (chi tiết hóa đơn, tổng tiền, tổng calo)
        """
        bill_details = []
        total_cost = 0
        total_calories = 0
        
        try:
            for item in food_items:
                # Tìm món ăn trong menu
                menu_item = self.menu[self.menu['item'] == item]
                
                if len(menu_item) > 0:
                    # Lấy thông tin của món ăn đầu tiên tìm thấy
                    price = menu_item.iloc[0]['price']
                    calories = menu_item.iloc[0]['calories']
                else:
                    # Nếu không tìm thấy, sử dụng giá trị mặc định
                    print(f"CẢNH BÁO: Không tìm thấy '{item}' trong menu. Sử dụng giá trị mặc định.")
                    price = 10000  # Giá mặc định: 10,000 VND
                    calories = 100  # Calo mặc định: 100 kcal
                
                # Thêm vào chi tiết hóa đơn
                bill_details.append({
                    'item': item,
                    'price': price,
                    'calories': calories
                })
                
                # Cập nhật tổng
                total_cost += price
                total_calories += calories
            
            return bill_details, total_cost, total_calories
            
        except Exception as e:
            print(f"Lỗi khi tính hóa đơn: {str(e)}")
            return [], 0, 0

if __name__ == "__main__":
    # Demo
    billing = BillingSystem()
    food_items = ['com', 'ga chien', 'canh chua', 'dua leo', 'dau hu']
    bill_details, total_cost, total_calories = billing.calculate_bill(food_items)
    
    print("Chi tiết hóa đơn:")
    for item in bill_details:
        print(f"{item['item']}: {item['price']} VND, {item['calories']} kcal")
        
    print(f"Tổng tiền: {total_cost} VND")
    print(f"Tổng calo: {total_calories} kcal")