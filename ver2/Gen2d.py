import os
import cv2
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# ======================================================================
# 1. TỐI ƯU HÓA GPU LOCAL (QUAN TRỌNG KHI CHẠY TRÊN MÁY NHÀ)
# Bật tính năng "Memory Growth" để TF chỉ ăn RAM vừa đủ, không nuốt trọn GPU
# ======================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Đã bật Memory Growth cho GPU Local! Sẵn sàng bào data an toàn.")
    except RuntimeError as e:
        print(e)

# ======================================================================
# 2. CẤU HÌNH ĐƯỜNG DẪN (Ổ D:\ CỦA ÔNG)
# ======================================================================
LATENT_DIM = 256 
WEIGHTS_PATH = r"D:\BraTS2021_Split_3Mask_V2\Models\CVAE\cvae_weights_best.weights.h5" 

TRAIN_MASK_DIR = r"D:\BraTS2021_Split_3Mask_V2\train\masks"
SYNTHETIC_IMG_DIR = r"D:\BraTS2021_Split_3Mask_V2\train_synthetic\images"
SYNTHETIC_MASK_DIR = r"D:\BraTS2021_Split_3Mask_V2\train_synthetic\masks"

os.makedirs(SYNTHETIC_IMG_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_MASK_DIR, exist_ok=True)

# ======================================================================
# 3. KIẾN TRÚC MẠNG (Bản chuẩn chống sọc, đã fix KL Loss)
# ======================================================================
class ConditionalVAE_Standard(tf.keras.Model):
    def __init__(self, latent_dim, **kwargs):
        super(ConditionalVAE_Standard, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(256, 256, 6)),
            tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(256, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ])

        self.decoder_z = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(16 * 16 * 256, activation="relu"),
            tf.keras.layers.Reshape((16, 16, 256)),
        ])
        
        self.decoder_up = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(16, 16, 259)),
            tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.Conv2D(3, 3, padding="same") 
        ])

    def encode(self, img, mask):
        x = tf.concat([img, mask], axis=-1)
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        logvar = tf.clip_by_value(logvar, -10.0, 10.0)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, mask):
        z_spatial = self.decoder_z(z)
        mask_small = tf.image.resize(mask, [16, 16], method='nearest')
        z_cond = tf.concat([z_spatial, mask_small], axis=-1)
        return self.decoder_up(z_cond)
    
    def call(self, inputs):
        img, mask = inputs
        mean, logvar = self.encode(img, mask)
        z = self.reparameterize(mean, logvar)
        return self.decode(z, mask)

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        img, mask = data
        BETA = 0.1 

        with tf.GradientTape() as tape:
            mean, logvar = self.encode(img, mask)
            z = self.reparameterize(mean, logvar)
            x_logit = self.decode(z, mask)

            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=img)
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

            # Đã fix lại KL Loss thành bản giải tích xịn nhất
            kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
            loss = -tf.reduce_mean(logpx_z - BETA * kl_loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        clipped_gradients = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in gradients]
        
        self.optimizer.apply_gradients(zip(clipped_gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

# ======================================================================
# 4. KHỞI TẠO VÀ LOAD TRỌNG SỐ
# ======================================================================
model = ConditionalVAE_Standard(LATENT_DIM)

print("⏳ Đang mồi TensorFlow Graph...")
dummy_img = tf.zeros((1, 256, 256, 3))
dummy_mask = tf.zeros((1, 256, 256, 3))
model([dummy_img, dummy_mask]) 

if not os.path.exists(WEIGHTS_PATH):
    print(f"❌ LỖI NGHIÊM TRỌNG: Không tìm thấy file weights tại {WEIGHTS_PATH}")
    print("👉 Ông check lại xem đã tải file weights từ Colab về máy tính chưa nhé!")
    exit()
else:
    model.load_weights(WEIGHTS_PATH, by_name=True, skip_mismatch=True)
    print(f"✅ Đã load weights thành công từ: {WEIGHTS_PATH}")

# ======================================================================
# 5. VÒNG LẶP SINH ẢNH ẢO
# ======================================================================
real_mask_files = [f for f in os.listdir(TRAIN_MASK_DIR) if f.endswith('.png') and not f.startswith('fake_')]
print(f"🚀 Bắt đầu sinh {len(real_mask_files)} ảnh não ảo từ Mask...")

for file_name in tqdm(real_mask_files):
    mask_path = os.path.join(TRAIN_MASK_DIR, file_name)
    
    mask = cv2.imread(mask_path)
    if mask is None:
        continue
        
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    mask_tensor = (mask_rgb / 255.0).astype(np.float32)
    mask_tensor = (mask_tensor > 0.5).astype(np.float32)
    mask_tensor_batch = np.expand_dims(mask_tensor, axis=0)
    
    # Random Z vector để mỗi lần sinh ra một vân não khác nhau
    z_random = tf.random.normal(shape=(1, LATENT_DIM))
    logits = model.decode(z_random, mask_tensor_batch)
    
    # Ép giá trị về [0, 1]
    fake_img_scaled = tf.sigmoid(logits).numpy()[0]
    
    # Trả về khoảng [0, 255] để lưu ảnh
    fake_img = (fake_img_scaled * 255).astype(np.uint8)
    fake_img_bgr = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)
    
    fake_file_name = f"fake_cvae_{file_name}"
    cv2.imwrite(os.path.join(SYNTHETIC_IMG_DIR, fake_file_name), fake_img_bgr)
    
    # Copy file mask đi kèm sang thư mục mới
    shutil.copy(mask_path, os.path.join(SYNTHETIC_MASK_DIR, fake_file_name))

print("🎉 Đã hoàn thành quá trình sinh ảnh ảo! Vào thư mục check hàng đi bro!")