import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from keras_unet_collection import models

# 1. BẬT MIXED PRECISION (Cứu VRAM RTX 4060)
mixed_precision.set_global_policy('mixed_float16')

# 2. HÀM TOÁN HỌC: DICE LOSS (Chuẩn theo Pseudo-code Bước 3)
def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    # Thêm 1e-5 để tránh lỗi chia cho 0
    return 1 - (numerator + 1e-5) / (denominator + 1e-5)

# 3. CHUẨN BỊ BĂNG CHUYỀN VÀ DATA
DIR_IMG_1 = r"D:\BraTS2021_Split\train\images" 
DIR_MASK_1 = r"D:\BraTS2021_Split\train\masks"

# 2. ĐƯỜNG DẪN CHUỒNG 2: ẢNH ẢO CVAE ĐẺ RA
DIR_IMG_2 = r"D:\BraTS2021_Split\train_synthetic\images" 
DIR_MASK_2 = r"D:\BraTS2021_Split\train_synthetic\masks"

# 3. LẤY DANH SÁCH FILE TỪ CẢ 2 CHUỒNG
img_list_1 = [os.path.join(DIR_IMG_1, f) for f in os.listdir(DIR_IMG_1)]
mask_list_1 = [os.path.join(DIR_MASK_1, f) for f in os.listdir(DIR_MASK_1)]

img_list_2 = [os.path.join(DIR_IMG_2, f) for f in os.listdir(DIR_IMG_2)]
mask_list_2 = [os.path.join(DIR_MASK_2, f) for f in os.listdir(DIR_MASK_2)]

img_files = np.array(sorted(img_list_1 + img_list_2))
mask_files = np.array(sorted(mask_list_1 + mask_list_2))

print(f"✅ Đã gộp thành công! Tổng cộng có {len(img_files)} cặp ảnh để đưa vào UNETR.")

def load_and_preprocess(img_path, mask_path):
    img = tf.cast(tf.image.decode_png(tf.io.read_file(img_path), channels=3), tf.float32) / 255.0
    mask = tf.cast(tf.image.decode_png(tf.io.read_file(mask_path), channels=1), tf.float32) / 255.0
    return img, mask

def create_dataset(x_paths, y_paths, batch_size=8, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((x_paths, y_paths))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(2000)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# =====================================================================
# 4. KHAI HỎA 5-FOLD CROSS VALIDATION
# =====================================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1

for train_index, val_index in kf.split(img_files):
    print(f"\n🚀 BẮT ĐẦU CHẠY FOLD SỐ {fold_no} / 5 ...")
    
    # Chia data cho Fold hiện tại
    train_x, val_x = img_files[train_index], img_files[val_index]
    train_y, val_y = mask_files[train_index], mask_files[val_index]
    #sửa batch size ở đây cho phù hợp với VRAM  nếu crash giảm xuống 8 hoặc 4
    train_dataset = create_dataset(train_x, train_y, batch_size=16, shuffle=True)
    val_dataset = create_dataset(val_x, val_y, batch_size=16, shuffle=False)

    # Khai báo lại Model cho mỗi Fold để không bị dính 'trí nhớ' của Fold cũ
    # 10 Attention heads theo Bảng 3
    unetr_model = models.transunet_2d(
        (256, 256, 3), 
        filter_num=[64, 128, 256, 512], 
        n_labels=1, 
        stack_num_down=2, 
        stack_num_up=2,
        embed_dim=768, 
        num_mlp=3072, 
        num_heads=10, 
        num_transformer=12, 
        activation='ReLU', 
        output_activation='Sigmoid',
        batch_norm=True
    )

    # Optimizer Adam lr=10^-5 theo Bảng 3
    optimizer = Adam(learning_rate=1e-5)
    unetr_model.compile(optimizer=optimizer, loss=dice_loss, metrics=['accuracy'])

    # Early Stopping sau 10 Epochs (Bước 6 Pseudo-code)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Lưu file model riêng cho từng Fold
    checkpoint = ModelCheckpoint(f"D:\\BraTS2021_2D\\unetr_fold_{fold_no}.h5", 
                                 monitor='val_loss', save_best_only=True)

    # Bắt đầu Train Fold hiện tại (Tác giả set 600, nhưng có Early Stop nên sẽ tự ngắt sớm)
    history = unetr_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=600,
        callbacks=[early_stop, checkpoint]
    )
    
    print(f"✅ Hoàn thành Fold {fold_no}!")
    
    # Xóa model khỏi RAM GPU để nhường chỗ cho Fold tiếp theo
    del unetr_model
    tf.keras.backend.clear_session()
    
    fold_no += 1

print("\n🎉 HOÀN TẤT 5-FOLD CROSS VALIDATION! AI ĐÃ TRỞ THÀNH PHÁP SƯ!")