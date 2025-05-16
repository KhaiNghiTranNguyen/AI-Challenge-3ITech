from src.detect import FoodDetector
from src.classify import FoodClassifier
from src.billing import BillingSystem

def main(image_path):
    class_names = [
        'banh mi', 'bap cai luoc', 'bap cai xao', 'bo xao', 'ca chien', 'ca chua', 'ca kho',
        'ca rot', 'canh bau', 'canh bi do', 'canh cai', 'canh chua', 'canh rong bien', 'chuoi',
        'com', 'dau bap', 'dau hu', 'dau que', 'do chua', 'dua hau', 'dua leo', 'ga chien',
        'ga kho', 'kho qua', 'kho tieu', 'kho trung', 'nuoc mam', 'nuoc tuong', 'oi', 'ot',
        'rau', 'rau muong', 'rau ngo', 'suon mieng', 'suon xao', 'thanh long', 'thit chien',
        'thit luoc', 'tom', 'trung chien', 'trung luoc'
    ]
    detector = FoodDetector()
    classifier = FoodClassifier(class_names=class_names)
    billing = BillingSystem()

    # Phát hiện và cắt ảnh món ăn sử dụng YOLO
    print("1. Bắt đầu phát hiện món ăn với YOLO...")
    cropped_paths, yolo_classes, results = detector.detect_and_crop(image_path)
    if len(cropped_paths) < 4:
        print("Lỗi: Phát hiện ít hơn 4 món ăn!")
        return

    print(f"Đã phát hiện {len(cropped_paths)} món ăn trong hình ảnh.")

    # Phân loại và so sánh
    print("\n2. Bắt đầu phân loại món ăn với CNN...")
    food_items = []
    for path, yolo_class in zip(cropped_paths, yolo_classes):
        cnn_class = classifier.classify(path)
        if cnn_class:
            food_items.append(cnn_class)
            print(f"Ảnh đã cắt: {path}, YOLO: {yolo_class}, CNN: {cnn_class}")
            if yolo_class != cnn_class:
                print(f"Phát hiện sự khác biệt, sử dụng lớp CNN: {cnn_class}")
        else:
            print(f"CNN thất bại cho {path}, sử dụng lớp YOLO: {yolo_class}")
            food_items.append(yolo_class)

    # Tính hóa đơn
    print("\n3. Tính toán hóa đơn...")
    bill_details, total_cost, total_calories = billing.calculate_bill(food_items)

    # In kết quả
    print("\nCác món ăn được phát hiện:")
    for detail in bill_details:
        print(f"Món: {detail['item']}, Giá: {detail['price']} VND, Calo: {detail['calories']} kcal")
    
    print(f"\nTổng tiền: {total_cost} VND")
    print(f"Tổng calo: {total_calories} kcal")

if __name__ == "__main__":
    image_path = "D:\khay_com\Tom rim me, dau hu nhoi, canh bi dao.png"
    main(image_path)