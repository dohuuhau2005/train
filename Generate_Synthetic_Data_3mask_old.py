import os
import cv2
import shutil # Kéo lên đây cho gọn
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tqdm import tqdm

# ======================================================================
# 1. KHAI BÁO LẠI KHUNG XƯƠNG CVAE (Để load file Weights)
# ======================================================================
LATENT_DIM = 100
DROPOUT_RATE = 0.2

encoder_inputs = layers.Input(shape=(256, 256, 3))
x = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Dropout(DROPOUT_RATE)(x)
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Dropout(DROPOUT_RATE)(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(DROPOUT_RATE)(x)
z_mean = layers.Dense(LATENT_DIM, dtype='float32')(x)
z_log_var = layers.Dense(LATENT_DIM, dtype='float32')(x)

# Đã Fix chuẩn dtype để không bị lỗi lúc Predict
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim), dtype=z_mean.dtype)
        half = tf.cast(0.5, z_log_var.dtype)
        return z_mean + tf.exp(half * z_log_var) * epsilon

z = Sampling()([z_mean, z_log_var])
encoder = Model(encoder_inputs, [z_mean, z_log_var, z])

latent_inputs = layers.Input(shape=(LATENT_DIM,))
x = layers.Dense(64 * 64 * 32, activation="relu")(latent_inputs)
x = layers.Dropout(DROPOUT_RATE)(x)
x = layers.Reshape((64, 64, 32))(x)
x = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Dropout(DROPOUT_RATE)(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", strides=2, padding="same", dtype='float32')(x)
decoder = Model(latent_inputs, decoder_outputs)

# Load Weights
encoder.load_weights(r"D:\BraTS2021_2D\cvae_weights_final.h5", by_name=True)
decoder.load_weights(r"D:\BraTS2021_2D\cvae_weights_final.h5", by_name=True)
print("✅ Đã gọi hồn CVAE thành công!")

# ======================================================================
# 2. TIẾN HÀNH "ĐẺ" ẢNH VÀ LƯU VÀO CHUỒNG SYNTHETIC
# ======================================================================
TRAIN_IMG_DIR = r"D:\BraTS2021_Split\train\images"
TRAIN_MASK_DIR = r"D:\BraTS2021_Split\train\masks"

SYNTHETIC_IMG_DIR = r"D:\BraTS2021_Split\train_synthetic\images"
SYNTHETIC_MASK_DIR = r"D:\BraTS2021_Split\train_synthetic\masks"
os.makedirs(SYNTHETIC_IMG_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_MASK_DIR, exist_ok=True)

real_img_files = [f for f in os.listdir(TRAIN_IMG_DIR) if not f.startswith('fake_')]

print(f"🚀 Bắt đầu sinh {len(real_img_files)} ảnh não ảo và copy Mask 3 lớp...")

for file_name in tqdm(real_img_files):
    img_path = os.path.join(TRAIN_IMG_DIR, file_name)
    mask_path = os.path.join(TRAIN_MASK_DIR, file_name)
    
    # Đọc và xử lý ảnh thật
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_scaled = np.expand_dims(img / 255.0, axis=0) 
    
    # Sinh ảnh ảo
    _, _, z_sampled = encoder.predict(img_scaled, verbose=0)
    fake_img_scaled = decoder.predict(z_sampled, verbose=0)[0]
    
    # Lưu file ảnh ảo
    fake_img = (fake_img_scaled * 255).astype(np.uint8)
    fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)
    fake_file_name = f"fake_cvae_{file_name}"
    cv2.imwrite(os.path.join(SYNTHETIC_IMG_DIR, fake_file_name), fake_img)
    
    # Copy nguyên vẹn file Mask 3 kênh gốc sang làm Mask cho ảnh ảo
    shutil.copy(mask_path, os.path.join(SYNTHETIC_MASK_DIR, fake_file_name))

print("🎉 TUYỆT VỜI! Đã có Data Ảo với Mask 3 loại (WT/TC/ET) nằm gọn trong thư mục synthetic!")