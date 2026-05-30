import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tqdm import tqdm

# ======================================================================
# 1. KHAI BÁO LẠI KHUNG XƯƠNG CVAE (Để load file Weights)
# ======================================================================
LATENT_DIM = 100
DROPOUT_RATE = 0.2

# Khai báo lại y chang file cũ để nạp linh hồn vào
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

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], tf.shape(z_mean)[1]))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

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

# Load Weights mà bro vừa cắm máy train
encoder.load_weights(r"D:\BraTS2021_2D\cvae_weights_final.h5", by_name=True)
decoder.load_weights(r"D:\BraTS2021_2D\cvae_weights_final.h5", by_name=True)
print("✅ Đã gọi hồn CVAE thành công!")

# ======================================================================
# 2. TIẾN HÀNH "ĐẺ" ẢNH VÀ TRỘN VÀO TẬP TRAIN
# ======================================================================
TRAIN_IMG_DIR = r"D:\BraTS2021_Split\train\images"
TRAIN_MASK_DIR = r"D:\BraTS2021_Split\train\masks"

SYNTHETIC_IMG_DIR = r"D:\BraTS2021_Split\train_synthetic\images"
SYNTHETIC_MASK_DIR = r"D:\BraTS2021_Split\train_synthetic\masks"
os.makedirs(SYNTHETIC_IMG_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_MASK_DIR, exist_ok=True)
# Lấy danh sách ảnh thật hiện có
real_img_files = [f for f in os.listdir(TRAIN_IMG_DIR) if not f.startswith('fake_')]

print(f"🚀 Bắt đầu sinh {len(real_img_files)} ảnh ảo từ ảnh thật...")

for file_name in tqdm(real_img_files):
    img_path = os.path.join(TRAIN_IMG_DIR, file_name)
    mask_path = os.path.join(TRAIN_MASK_DIR, file_name)
    
    # 1. Đọc ảnh thật
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_scaled = np.expand_dims(img / 255.0, axis=0) # Shape: (1, 256, 256, 3)
    
    # 2. Đưa qua Encoder để nén và cộng nhiễu -> Lấy Z
    _, _, z_sampled = encoder.predict(img_scaled, verbose=0)
    
    # 3. Đưa qua Decoder để giải nén ra ảnh ảo
    fake_img_scaled = decoder.predict(z_sampled, verbose=0)[0]
    
    # Chuyển hệ màu về lưu file
    fake_img = (fake_img_scaled * 255).astype(np.uint8)
    fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)
    
    # 4. LƯU ẢNH ẢO VÀ NHÂN BẢN MASK GỐC (Bí thuật ở đây)
    fake_file_name = f"fake_cvae_{file_name}"
    
    # Lưu ảnh ảo vào chung chuồng Train/Images
    cv2.imwrite(os.path.join(SYNTHETIC_IMG_DIR, fake_file_name), fake_img)
    
    # Copy cái Mask gốc, đổi tên thành fake_... rồi lưu vào chuồng Train/Masks
    import shutil
    shutil.copy(mask_path, os.path.join(SYNTHETIC_MASK_DIR, fake_file_name))

print("🎉 TUYỆT VỜI! Tập Train đã được nhân đôi sức mạnh. Sẵn sàng cho UNETR càn quét!")