import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import albumentations as A
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision

# ======================================================================   epoch23
# 1. CẤU HÌNH CƠ BẢN & ĐƯỜNG DẪN
# ======================================================================
mixed_precision.set_global_policy('mixed_float16')
print("✅ Đã bật Mixed Precision. Card RTX 4060 tiếp tục chạy max ping!")

# 🚨 CHÚ Ý: ĐIỀN EPOCH ÔNG MUỐN CHẠY TIẾP VÀO ĐÂY
# Ví dụ: file đang load là "advanced_backup_epoch_10.keras" thì để INITIAL_EPOCH = 10
INITIAL_EPOCH = 77
TOTAL_EPOCHS = 250 # Tổng số epoch muốn train (tính cả cũ lẫn mới)

# ĐƯỜNG DẪN TỚI FILE MODEL ĐANG TRAIN DỞ (Sửa lại tên file cho đúng với mốc ông muốn load)
RESUME_MODEL_PATH = r"D:\BraTS2021_Split_3Mask_V2\Models\attention_unet_advanced_best.keras" 

# Đường dẫn data (Giữ nguyên y hệt lúc Train)
TRAIN_IMG_REAL = r"D:\BraTS2021_Split_3Mask_V2\train\images"
TRAIN_MASK_REAL = r"D:\BraTS2021_Split_3Mask_V2\train\masks"
TRAIN_IMG_FAKE = r"" 
TRAIN_MASK_FAKE = r""
VAL_IMG_DIR = r"D:\BraTS2021_Split_3Mask_V2\val\images"
VAL_MASK_DIR = r"D:\BraTS2021_Split_3Mask_V2\val\masks"

CKPT_DIR = r"D:\BraTS2021_Split_3Mask_V2\Models"
CKPT_PATH = os.path.join(CKPT_DIR, "attention_unet_advanced_best.keras")
CSV_PATH = os.path.join(CKPT_DIR, "advanced_training_history.csv")

# ======================================================================
# 2. BĂNG CHUYỀN DATA (Giữ nguyên 100%)
# ======================================================================
BATCH_SIZE = 16 

train_img_paths = sorted(glob.glob(os.path.join(TRAIN_IMG_REAL, "*.png")))
if os.path.exists(TRAIN_IMG_FAKE):
    train_img_paths += sorted(glob.glob(os.path.join(TRAIN_IMG_FAKE, "*.png")))

train_mask_paths = sorted(glob.glob(os.path.join(TRAIN_MASK_REAL, "*.png")))
if os.path.exists(TRAIN_MASK_FAKE):
    train_mask_paths += sorted(glob.glob(os.path.join(TRAIN_MASK_FAKE, "*.png")))

val_img_paths = sorted(glob.glob(os.path.join(VAL_IMG_DIR, "*.png")))
val_mask_paths = sorted(glob.glob(os.path.join(VAL_MASK_DIR, "*.png")))

aug_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5)
])

def parse_function(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.cast(mask > 0.5, tf.float32) 
    return img, mask

def albumentations_fn(img, mask):
    augmented = aug_pipeline(image=img.numpy(), mask=mask.numpy())
    return augmented['image'], augmented['mask']

def augment_data(img, mask):
    aug_img, aug_mask = tf.py_function(func=albumentations_fn, inp=[img, mask], Tout=[tf.float32, tf.float32])
    aug_img.set_shape([256, 256, 3])
    aug_mask.set_shape([256, 256, 3])
    aug_img = tf.clip_by_value(aug_img, 0.0, 1.0)
    return aug_img, aug_mask

def get_dataset(img_paths, mask_paths, batch_size, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=2000, reshuffle_each_iteration=True)
    ds = ds.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = get_dataset(train_img_paths, train_mask_paths, BATCH_SIZE, shuffle=True, augment=True)
val_ds = get_dataset(val_img_paths, val_mask_paths, BATCH_SIZE, shuffle=False, augment=False)

# ======================================================================
# 3. HÀM LOSS VÀ METRICS CUSTOM (BẮT BUỘC PHẢI CÓ ĐỂ LOAD MODEL)
# ======================================================================
def dice_per_channel(y_true, y_pred, smooth=1e-6):
    dices = []
    for c in range(3):
        yt = tf.reshape(y_true[..., c], [-1])
        yp = tf.reshape(y_pred[..., c], [-1])
        inter = tf.reduce_sum(yt * yp)
        dice = (2. * inter + smooth) / (tf.reduce_sum(yt) + tf.reduce_sum(yp) + smooth)
        dices.append(dice)
    return tf.reduce_mean(dices)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_per_channel(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

def dice_coef(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    return dice_per_channel(y_true, y_pred)

# ======================================================================
# 4. LOAD MODEL TỪ ĐIỂM DỪNG
# ======================================================================
print(f"🔄 Đang đánh thức mô hình từ giấc ngủ tại: {RESUME_MODEL_PATH}...")
model = tf.keras.models.load_model(
    RESUME_MODEL_PATH,
    custom_objects={
        'bce_dice_loss': bce_dice_loss,
        'dice_coef': dice_coef,
        'dice_loss': dice_loss,
        'dice_per_channel': dice_per_channel
    }
)
print("✅ Đã load Model và Optimizer thành công! Tiếp tục cuộc chiến!")

# ======================================================================
# 5. CẤU HÌNH CALLBACK VÀ CHẠY TIẾP
# ======================================================================
class BackupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            backup_path = os.path.join(CKPT_DIR, f"advanced_backup_epoch_{epoch+1}.keras")
            self.model.save(backup_path)
            print(f"\n[AUTO BACKUP] Đã lưu mốc an toàn tại Epoch {epoch+1}!")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        CKPT_PATH, monitor="val_dice_coef", mode="max", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_dice_coef", mode="max", patience=15, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_dice_coef", mode="max", factor=0.5, patience=4, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.CSVLogger(
        CSV_PATH, separator=",", 
        append=True # 🚨 QUAN TRỌNG: Đổi thành True để nó viết tiếp vào file CSV cũ, không bị xóa đè
    ),
    BackupCallback()
]

print(f"🚀 Bắt đầu Train nối tiếp từ Epoch {INITIAL_EPOCH + 1}...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=INITIAL_EPOCH, # 🚨 QUAN TRỌNG: Khai báo epoch bắt đầu để thanh tiến trình và CSV chạy đúng
    epochs=TOTAL_EPOCHS, 
    callbacks=callbacks
)

print(f"🎉 Hoàn tất quá trình Resume Training! Model lưu tại: {CKPT_DIR}")