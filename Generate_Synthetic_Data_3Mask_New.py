import os
import cv2
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# ======================================================================
# 1. BÊ NGUYÊN KIẾN TRÚC MẠNG BẢN MỚI NHẤT VÀO
# ======================================================================
LATENT_DIM = 256 # Bản mới não bự hơn, nhớ đổi số này!

class CVAE_TF_Standard(tf.keras.Model):
    def __init__(self, latent_dim, **kwargs):
        super(CVAE_TF_Standard, self).__init__(**kwargs)
        self.latent_dim = latent_dim

        # --- ENCODER ---
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(256, 256, 3)),
            tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(256, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ])

        # --- DECODER ---
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(16 * 16 * 256, activation="relu"),
            tf.keras.layers.Reshape((16, 16, 256)),
            tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding="same"), # Xuất Logits
        ])
    def call(self, inputs):
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)
        return self.decode(z)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        logvar = tf.clip_by_value(logvar, -10.0, 10.0)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z):
        return self.decoder(z)

# Khởi tạo model
model = CVAE_TF_Standard(LATENT_DIM)

# MẸO: Chạy nháp 1 bức ảnh rỗng để TensorFlow dựng khung (Build) trước khi load weights
dummy_input = tf.zeros((1, 256, 256, 3))
model(dummy_input)

# LOAD WEIGHTS (🔴 Nhớ sửa lại tên file h5 bro đang có nhé, ví dụ epoch 5)
WEIGHTS_PATH = r"D:\BraTS2021_2D\cvae_weights_resumed_final.h5" 
model.load_weights(WEIGHTS_PATH)
print(f"✅ Đã gọi hồn nội công từ file: {WEIGHTS_PATH}")

# ======================================================================
# 2. ĐƯỜNG DẪN DỮ LIỆU
# ======================================================================
TRAIN_IMG_DIR = r"D:\BraTS2021_Split_3Mask\train\images"
TRAIN_MASK_DIR = r"D:\BraTS2021_Split_3Mask\train\masks"

SYNTHETIC_IMG_DIR = r"D:\BraTS2021_Split_3Mask\train_synthetic\images"
SYNTHETIC_MASK_DIR = r"D:\BraTS2021_Split_3Mask\train_synthetic\masks"
os.makedirs(SYNTHETIC_IMG_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_MASK_DIR, exist_ok=True)

real_img_files = [f for f in os.listdir(TRAIN_IMG_DIR) if not f.startswith('fake_')]
print(f"🚀 Bắt đầu sinh {len(real_img_files)} ảnh ảo từ ảnh thật...")

for file_name in tqdm(real_img_files):
    img_path = os.path.join(TRAIN_IMG_DIR, file_name)
    mask_path = os.path.join(TRAIN_MASK_DIR, file_name)
    
    # Đọc ảnh thật
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_scaled = np.expand_dims(img / 255.0, axis=0).astype(np.float32)
    
    # Nén -> Lấy Z -> Giải nén
    mean, logvar = model.encode(img_scaled)
    z_sampled = model.reparameterize(mean, logvar)
    logits = model.decode(z_sampled)
    
    # 🚨 BÍ QUYẾT TRÁNH MÀU XÁM KHI XUẤT ẢNH: BỌC HÀM SIGMOID Ở ĐÂY
    fake_img_scaled = tf.sigmoid(logits).numpy()[0]
    
    # Chuyển hệ màu về lưu file
    fake_img = (fake_img_scaled * 255).astype(np.uint8)
    fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)
    
    # LƯU ẢNH VÀ COPY MASK
    fake_file_name = f"fake_cvae_{file_name}"
    cv2.imwrite(os.path.join(SYNTHETIC_IMG_DIR, fake_file_name), fake_img)
    shutil.copy(mask_path, os.path.join(SYNTHETIC_MASK_DIR, fake_file_name))

print("🎉 TUYỆT VỜI! Đã rặn đẻ xong bộ Data xịn xò 3 Mask!")