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
# Gộp cả ảnh thật và ảnh ảo CVAE đẻ ra để Train cho model khôn nhất
TRAIN_IMG_REAL = r"D:\BraTS2021_Split_3Mask\train\images"
TRAIN_MASK_REAL = r"D:\BraTS2021_Split_3Mask\train\masks"

TRAIN_IMG_FAKE = r"D:\BraTS2021_Split_3Mask\train_synthetic\images"
TRAIN_MASK_FAKE = r"D:\BraTS2021_Split_3Mask\train_synthetic\masks"

VAL_IMG_DIR = r"D:\BraTS2021_Split_3Mask\val\images"
VAL_MASK_DIR = r"D:\BraTS2021_Split_3Mask\val\masks"

# Thư mục lưu Model
CKPT_DIR = r"D:\BraTS2021_Split_3Mask\Models"
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(CKPT_DIR, "attention_unet_wt_tc_et_best.keras")

# ======================================================================
# 2. XÂY DỰNG BĂNG CHUYỀN DATA (DATA LOADER)
# ======================================================================
BATCH_SIZE = 8 # Hạ xuống 8 để an toàn cho card 8GB (Giáo viên xài A100 40GB mới dám để to)

# Lấy danh sách file (Nhớ sort để ảnh và mask khớp nhau 100%)
train_img_paths = sorted(glob.glob(os.path.join(TRAIN_IMG_REAL, "*.png"))) + \
                  sorted(glob.glob(os.path.join(TRAIN_IMG_FAKE, "*.png")))
train_mask_paths = sorted(glob.glob(os.path.join(TRAIN_MASK_REAL, "*.png"))) + \
                   sorted(glob.glob(os.path.join(TRAIN_MASK_FAKE, "*.png")))

val_img_paths = sorted(glob.glob(os.path.join(VAL_IMG_DIR, "*.png")))
val_mask_paths = sorted(glob.glob(os.path.join(VAL_MASK_DIR, "*.png")))

def parse_function(img_path, mask_path):
    # Đọc ảnh (3 kênh MRI: FLAIR, T2, T1ce)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0

    # Đọc mask (3 kênh Mask: WT, TC, ET)
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.cast(mask > 0.5, tf.float32) # Đảm bảo mask là nhị phân 0 hoặc 1
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

print(f"✅ Train batches: {len(train_ds)} (Gồm cả ảnh thật + ảo CVAE)")
print(f"✅ Val batches  : {len(val_ds)}")

# ======================================================================
# 3. MÔ HÌNH ATTENTION U-NET (CODE CỦA GIÁO VIÊN)
# ======================================================================
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def attention_gate(x, g, filters):
    theta_x = layers.Conv2D(filters, 1)(x)
    phi_g   = layers.Conv2D(filters, 1)(g)

    add = layers.Add()([theta_x, phi_g])
    add = layers.Activation("relu")(add)

    psi = layers.Conv2D(filters, 1, activation="sigmoid")(add)
    return layers.Multiply()([x, psi])

def build_attention_unet(input_shape=(256,256,3), n_classes=3):
    inputs = layers.Input(input_shape)

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 256)
    p4 = layers.MaxPooling2D()(c4)

    bn = conv_block(p4, 512)

    u1 = layers.UpSampling2D()(bn)
    a1 = attention_gate(c4, u1, 256)
    u1 = layers.Concatenate()([u1, a1])
    c5 = conv_block(u1, 256)

    u2 = layers.UpSampling2D()(c5)
    a2 = attention_gate(c3, u2, 128)
    u2 = layers.Concatenate()([u2, a2])
    c6 = conv_block(u2, 128)

    u3 = layers.UpSampling2D()(c6)
    a3 = attention_gate(c2, u3, 64)
    u3 = layers.Concatenate()([u3, a3])
    c7 = conv_block(u3, 64)

    u4 = layers.UpSampling2D()(c7)
    a4 = attention_gate(c1, u4, 32)
    u4 = layers.Concatenate()([u4, a4])
    c8 = conv_block(u4, 32)

    # Đã thêm dtype='float32' để tương thích với Mixed Precision
    outputs = layers.Conv2D(n_classes, 1, activation="sigmoid", dtype='float32')(c8)

    return models.Model(inputs, outputs)

# ======================================================================
# 4. HÀM LOSS VÀ METRICS ĐA KÊNH (MULTI-HEAD BCE + DICE)
# ======================================================================
def dice_per_channel(y_true, y_pred, smooth=1e-6):
    dices = []
    for c in range(3):
        yt = tf.reshape(y_true[..., c], [-1])
        yp = tf.reshape(y_pred[..., c], [-1])

        inter = tf.reduce_sum(yt * yp)
        dice = (2. * inter + smooth) / (
            tf.reduce_sum(yt) + tf.reduce_sum(yp) + smooth
        )
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
# 5. KHỞI TẠO VÀ BẮT ĐẦU HUẤN LUYỆN
# ======================================================================
model = build_attention_unet(input_shape=(256,256,3), n_classes=3)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=bce_dice_loss,
    metrics=[dice_coef]
)

# 🛡️ BÙA HỘ MỆNH: AUTO BACKUP MỖI 10 EPOCH
class BackupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            backup_path = os.path.join(CKPT_DIR, f"attention_unet_backup_epoch_{epoch+1}.keras")
            self.model.save(backup_path)
            print(f"\n💾 [AUTO BACKUP] Đã lưu mốc an toàn tại Epoch {epoch+1}!")

backup_10_epochs = BackupCallback()

# Callbacks cấu hình
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        CKPT_PATH,
        monitor="val_dice_coef",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_dice_coef",
        mode="max",
        patience=10, # 10 epoch không tiến bộ sẽ dừng
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_dice_coef",
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    ),
    backup_10_epochs # Nhét bùa hộ mệnh vào đây
]

print("🚀 Khai hỏa quá trình Train Attention U-Net (Auto Backup 10 Epochs)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100, # Cứ để 100, EarlyStopping sẽ tự ngắt khi nó đạt đỉnh (tầm 30-40)
    callbacks=callbacks
)

# Lưu lịch sử Train ra file CSV để sau này vẽ biểu đồ
hist_df = pd.DataFrame(history.history)
csv_path = os.path.join(CKPT_DIR, "training_history.csv")
hist_df.to_csv(csv_path, index=False)

print(f"🎉 Training hoàn tất! Model và Lịch sử đã được lưu tại: {CKPT_DIR}")