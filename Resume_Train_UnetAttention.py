import os
import glob
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision

# ======================================================================
# MA PHÁP: BẬT MIXED PRECISION ĐỂ TĂNG TỐC VÀ TIẾT KIỆM VRAM
# ======================================================================
mixed_precision.set_global_policy('mixed_float16')
print("✅ Đã bật Mixed Precision. Card RTX 4060 sẽ chạy hết tốc lực!")

# ======================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU
# ======================================================================
TRAIN_IMG_REAL = r"D:\BraTS2021_Split_3Mask\train\images"
TRAIN_MASK_REAL = r"D:\BraTS2021_Split_3Mask\train\masks"
TRAIN_IMG_FAKE = r"D:\BraTS2021_Split_3Mask\train_synthetic\images"
TRAIN_MASK_FAKE = r"D:\BraTS2021_Split_3Mask\train_synthetic\masks"
VAL_IMG_DIR = r"D:\BraTS2021_Split_3Mask\val\images"
VAL_MASK_DIR = r"D:\BraTS2021_Split_3Mask\val\masks"

CKPT_DIR = r"D:\BraTS2021_Split_3Mask\Models"
os.makedirs(CKPT_DIR, exist_ok=True)

# THAY ĐỔI: Tên file model CŨ để load lên và tên file MỚI để lưu
OLD_MODEL_PATH = os.path.join(CKPT_DIR, "attention_unet_wt_tc_et_best.keras")
NEW_MODEL_PATH = os.path.join(CKPT_DIR, "attention_unet_wt_tc_et_resumed_best.keras")

# ======================================================================
# 2. XÂY DỰNG BĂNG CHUYỀN DATA (DATA LOADER)
# ======================================================================
BATCH_SIZE = 8

train_img_paths = sorted(glob.glob(os.path.join(TRAIN_IMG_REAL, "*.png"))) + \
                  sorted(glob.glob(os.path.join(TRAIN_IMG_FAKE, "*.png")))
train_mask_paths = sorted(glob.glob(os.path.join(TRAIN_MASK_REAL, "*.png"))) + \
                   sorted(glob.glob(os.path.join(TRAIN_MASK_FAKE, "*.png")))
val_img_paths = sorted(glob.glob(os.path.join(VAL_IMG_DIR, "*.png")))
val_mask_paths = sorted(glob.glob(os.path.join(VAL_MASK_DIR, "*.png")))

def parse_function(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.cast(mask > 0.5, tf.float32)
    return img, mask

def get_dataset(img_paths, mask_paths, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    ds = ds.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = get_dataset(train_img_paths, train_mask_paths, BATCH_SIZE, shuffle=True)
val_ds = get_dataset(val_img_paths, val_mask_paths, BATCH_SIZE, shuffle=False)

print(f"✅ Train batches: {len(train_ds)}")
print(f"✅ Val batches  : {len(val_ds)}")

# ======================================================================
# 3. KHAI BÁO HÀM CUSTOM ĐỂ LOAD MODEL
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
# 4. LOAD MODEL CŨ VÀ BẮT ĐẦU HUẤN LUYỆN TIẾP (RESUME)
# ======================================================================
if not os.path.exists(OLD_MODEL_PATH):
    print(f"❌ KHÔNG TÌM THẤY FILE MODEL CŨ TẠI: {OLD_MODEL_PATH}")
    exit()

print(f"🔄 Đang Load Model cũ từ: {OLD_MODEL_PATH}...")
# Không gọi hàm build_attention_unet nữa, load thẳng model đã train
model = tf.keras.models.load_model(
    OLD_MODEL_PATH, 
    custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef}
)
print("✅ Load Model thành công!")

# THAY ĐỔI QUAN TRỌNG: Compile lại model với Learning Rate CỰC NHỎ (1e-5) để tỉa tót
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5), # Nhỏ hơn 10 lần so với lúc đầu (1e-4)
    loss=bce_dice_loss,
    metrics=[dice_coef]
)

# 🛡️ BÙA HỘ MỆNH
class BackupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            backup_path = os.path.join(CKPT_DIR, f"attention_unet_resumed_backup_epoch_{epoch+1}.keras")
            self.model.save(backup_path)
            print(f"\n💾 [AUTO BACKUP] Đã lưu mốc an toàn tại Epoch {epoch+1}!")

backup_10_epochs = BackupCallback()

# Callbacks cấu hình lưu tên file MỚI để không ghi đè file cũ
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        NEW_MODEL_PATH,
        monitor="val_dice_coef",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_dice_coef",
        mode="max",
        patience=10, 
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_dice_coef",
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-7, # Cho phép min_lr nhỏ hơn nữa
        verbose=1
    ),
    backup_10_epochs
]

print("\n🚀 BẮT ĐẦU FINE-TUNING (RESUME TRAINING) TỪ FILE BEST.KERAS CŨ...")
# Chạy thêm 50 Epochs nữa xem sao
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50, 
    callbacks=callbacks
)

# Lưu lịch sử Train Resume ra file CSV
hist_df = pd.DataFrame(history.history)
csv_path = os.path.join(CKPT_DIR, "resumed_training_history.csv")
hist_df.to_csv(csv_path, index=False)

print(f"🎉 Fine-tuning hoàn tất! Model MỚI đã được lưu tại: {NEW_MODEL_PATH}")