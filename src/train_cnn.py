import os
import numpy as np
import cv2
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Paths
TRAIN_DIR = "data/classification_dataset_augmented/train"  # Thư mục chứa dữ liệu huấn luyện
VAL_DIR = "data/classification_dataset_augmented/val"     # Thư mục chứa dữ liệu validation  
MODEL_PATH = "models/cnn.h5"  # Đường dẫn lưu mô hình
IMG_SIZE = (224, 224)  # Kích thước ảnh
BATCH_SIZE = 16  # Batch size
EPOCHS = 100

def check_images_in_directory(directory):
    print(f"Kiểm tra ảnh trong thư mục {directory}...")
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    valid_images = []
    invalid_images = []
    class_samples = {}
    
    try:
        # Kiểm tra xem đường dẫn có tồn tại không
        if not os.path.exists(directory):
            print(f"Thư mục {directory} không tồn tại!")
            return [], [], {}
            
        # Nếu là thư mục chứa các thư mục con (mỗi thư mục con là một class)
        for class_name in os.listdir(directory):
            class_dir = os.path.join(directory, class_name)
            if os.path.isdir(class_dir):
                class_samples[class_name] = 0
                for file in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, file)
                    if os.path.isfile(file_path):
                        ext = os.path.splitext(file)[1].lower()
                        if ext in valid_extensions:
                            try:
                                img = cv2.imread(file_path)
                                if img is None:
                                    invalid_images.append(file_path)
                                else:
                                    valid_images.append(file_path)
                                    class_samples[class_name] += 1
                            except Exception as e:
                                print(f"Lỗi khi đọc ảnh {file_path}: {e}")
                                invalid_images.append(file_path)
        
        # Hiển thị thống kê số lượng ảnh cho mỗi class
        print("\nThống kê số lượng ảnh cho mỗi class:")
        for class_name, count in class_samples.items():
            print(f"{class_name}: {count} ảnh")
                
    except Exception as e:
        print(f"Lỗi khi đọc thư mục {directory}: {e}")
        return [], [], {}
    
    print(f"\nTổng số ảnh hợp lệ: {len(valid_images)}")
    print(f"Tổng số ảnh không đọc được: {len(invalid_images)}")
    print(f"Tổng số classes: {len(class_samples)}")
    
    return valid_images, invalid_images, class_samples

# Kiểm tra ảnh trong thư mục huấn luyện và validation
print("Kiểm tra dữ liệu huấn luyện...")
valid_train_images, invalid_train_images, train_class_samples = check_images_in_directory(TRAIN_DIR)

print("\nKiểm tra dữ liệu validation...")
valid_val_images, invalid_val_images, val_class_samples = check_images_in_directory(VAL_DIR)

# Lấy danh sách các class từ cấu trúc thư mục
train_classes = list(train_class_samples.keys())
val_classes = list(val_class_samples.keys())

print(f"Tìm thấy {len(train_classes)} classes trong tập huấn luyện: {train_classes}")
print(f"Tìm thấy {len(val_classes)} classes trong tập validation: {val_classes}")

# Kiểm tra xem tất cả các lớp trong tập validation có trong tập huấn luyện không
missing_classes = [cls for cls in val_classes if cls not in train_classes]
if missing_classes:
    print(f"CẢNH BÁO: Các lớp sau có trong tập validation nhưng không có trong tập huấn luyện: {missing_classes}")

# Nếu không có ảnh hợp lệ, dừng chương trình
if len(valid_train_images) == 0:
    print("CẢNH BÁO: Không tìm thấy ảnh hợp lệ nào trong thư mục huấn luyện!")
    exit()
if len(valid_val_images) == 0:
    print("CẢNH BÁO: Không tìm thấy ảnh hợp lệ nào trong thư mục validation!")
    exit()

# 1. Chuẩn bị dữ liệu với tăng cường dữ liệu mạnh mẽ cho tập huấn luyện
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,  # Thêm đảo ngược dọc
    brightness_range=[0.7, 1.3],
    channel_shift_range=0.2,
    fill_mode='nearest'
)

# Validation data chỉ cần rescale
validation_datagen = ImageDataGenerator(
    rescale=1./255
)

# Tải dữ liệu huấn luyện
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Tạo validation generator từ tập validation riêng
validation_generator = validation_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Không shuffle validation để đánh giá nhất quán
)

# Lấy số lượng classes từ generator để đảm bảo khớp với dữ liệu
num_classes = len(train_generator.class_indices)
print(f"Số lượng classes từ generator huấn luyện: {num_classes}")
print(f"Classes từ generator huấn luyện: {train_generator.class_indices}")

val_num_classes = len(validation_generator.class_indices)
print(f"Số lượng classes từ generator validation: {val_num_classes}")
print(f"Classes từ generator validation: {validation_generator.class_indices}")

# Kiểm tra xem các tập huấn luyện và validation có cùng số lượng lớp không
if num_classes != val_num_classes:
    print(f"CẢNH BÁO: Số lượng classes khác nhau giữa tập huấn luyện ({num_classes}) và tập validation ({val_num_classes})!")

# 2. Xây dựng mô hình CNN cải tiến với kết nối tắt
def create_improved_cnn_model(input_shape, num_classes):
    """
    Tạo một mô hình CNN cải tiến với kết nối tắt để giúp gradient flow tốt hơn
    """
    inputs = Input(shape=input_shape)
    
    # Block đầu tiên: Conv -> BN -> ReLU -> MaxPool
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_regularizer=l2(0.0001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Function để tạo khối kết nối tắt
    def improved_block(x, filters, strides=1):
        """
        Tạo một khối có kết nối tắt để cải thiện việc truyền gradient
        """
        # Lưu đầu vào cho kết nối tắt
        shortcut = x
        
        # Nhánh chính: BN -> ReLU -> Conv -> BN -> ReLU -> Conv (kiểu pre-activation)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3, 3), strides=strides, padding='same', kernel_regularizer=l2(0.0001))(x)
        
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(x)
        
        # Nếu kích thước thay đổi, cần điều chỉnh shortcut bằng Conv 1x1
        if strides > 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding='same', kernel_regularizer=l2(0.0001))(shortcut)
        
        # Cộng đầu ra của nhánh chính với shortcut
        x = layers.add([x, shortcut])
        return x
    
    # Các giai đoạn (stage) với khối cải tiến, tăng dần số bộ lọc
    # Giai đoạn 1: 64 filters
    x = improved_block(x, 64)
    x = improved_block(x, 64)
    
    # Giai đoạn 2: 128 filters
    x = improved_block(x, 128, strides=2)
    x = improved_block(x, 128)
    
    # Giai đoạn 3: 256 filters
    x = improved_block(x, 256, strides=2)
    x = improved_block(x, 256)
    
    # Giai đoạn 4: 512 filters
    x = improved_block(x, 512, strides=2)
    x = improved_block(x, 512)
    
    # Global Average Pooling và Classification
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)  # Dropout để giảm overfitting
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# Tạo mô hình
model = create_improved_cnn_model((224, 224, 3), num_classes)

# Biên dịch mô hình
from tensorflow.keras.optimizers import Adam
model.compile(
    optimizer=Adam(learning_rate=0.0005),  # Learning rate thấp hơn để hội tụ tốt hơn
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Tóm tắt mô hình
model.summary()

# 3. Callbacks để tránh overfitting
# Đảm bảo thư mục tồn tại để lưu mô hình
os.makedirs(os.path.dirname(MODEL_PATH) if os.path.dirname(MODEL_PATH) else '.', exist_ok=True)

# Early stopping để dừng khi validation loss không giảm
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=40,  # Tăng patience để cho mô hình thời gian hội tụ
    restore_best_weights=True,
    verbose=1
)

# Lưu model có val_accuracy tốt nhất
checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Giảm learning rate khi plateau với hệ số nhỏ hơn
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,  # Giảm 80% learning rate
    patience=6,
    min_lr=1e-7,
    verbose=1
)

# 4. Huấn luyện mô hình
try:
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[early_stopping, checkpoint, reduce_lr]
    )

    # 5. Đánh giá và phân tích overfitting
    plt.figure(figsize=(12, 5))

    # Tạo thư mục models/cnn nếu nó không tồn tại
    os.makedirs('models/cnn', exist_ok=True)

    # Đồ thị accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Đồ thị loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('models/cnn/training_history.png')
    plt.show()

    # Vẽ thêm đồ thị cho learning rate
    if 'lr' in history.history:
        plt.figure(figsize=(10, 4))
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.savefig('models/cnn/learning_rate_history.png')
        plt.show()

    # Lưu lịch sử huấn luyện vào file
    import pandas as pd
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('models/cnn/training_history_improved.csv', index=False)
    print("Đã lưu lịch sử huấn luyện vào models/cnn/training_history_improved.csv")

    # Kiểm tra overfitting qua chênh lệch giữa training và validation
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]

    print("\n=== PHÂN TÍCH OVERFITTING ===")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Chênh lệch accuracy: {(train_acc - val_acc):.4f}")
    print(f"Training loss: {train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Chênh lệch loss: {(val_loss - train_loss):.4f}")

    # Đánh giá mức độ overfitting
    if train_acc - val_acc > 0.1:
        overfitting_level = "Nghiêm trọng" if train_acc - val_acc > 0.2 else "Nhẹ"
        print(f"\nCẢNH BÁO: Phát hiện overfitting mức độ {overfitting_level}!")
        print("\nCác giải pháp cho overfitting:")
        print("1. Tăng cường dữ liệu (thử các tham số data augmentation khác)")
        print("2. Thu thập thêm dữ liệu cho tập huấn luyện")
        print("3. Tăng dropout hoặc thêm BatchNormalization")
        print("4. Tăng L2 regularization (hiện tại 0.0001)")
        print("5. Giảm số lớp hoặc số bộ lọc trong các lớp CNN")
    else:
        print("\nMô hình không có dấu hiệu overfitting nghiêm trọng.")

    # Lưu model cuối cùng
    model.save("models/cnn.h5")
    print("\nĐã lưu model cuối cùng tại models/cnn.h5")

except Exception as e:
    print(f"Lỗi trong quá trình huấn luyện: {e}")
    
    # In ra thông tin debug để giúp tìm lỗi
    print(f"\nThông tin debug:")
    print(f"Shape của đầu ra từ mô hình: {model.output_shape}")
    print(f"Số lượng classes từ generator huấn luyện: {len(train_generator.class_indices)}")
    print(f"Số lượng classes từ generator validation: {len(validation_generator.class_indices)}")
    
    # Kiểm tra một batch từ generator
    x_batch, y_batch = next(train_generator)
    print(f"Shape của X batch từ train: {x_batch.shape}")
    print(f"Shape của Y batch từ train: {y_batch.shape}")
    
    try:
        x_val_batch, y_val_batch = next(validation_generator)
        print(f"Shape của X batch từ validation: {x_val_batch.shape}")
        print(f"Shape của Y batch từ validation: {y_val_batch.shape}")
    except Exception as val_err:
        print(f"Lỗi khi lấy batch từ validation generator: {val_err}")