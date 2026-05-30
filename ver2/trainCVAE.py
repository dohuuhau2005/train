import os
import tensorflow as tf

# 🚨 TẮT HOÀN TOÀN MIXED PRECISION (Kẻ thù của VAE)
BATCH_SIZE = 16

IMG_DIR = r"D:\BraTS2021_Split_3Mask_V2\train\images" 
MASK_DIR = r"D:\BraTS2021_Split_3Mask_V2\train\masks" 

SAVE_DIR = r"D:\BraTS2021_Split_3Mask_V2\Models\CVAE"
os.makedirs(SAVE_DIR, exist_ok=True)
CSV_PATH = os.path.join(SAVE_DIR, "cvae_training_history.csv")
BEST_WEIGHTS_PATH = os.path.join(SAVE_DIR, "cvae_weights_best.weights.h5")

all_image_paths = sorted([os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith('.png')])
all_mask_paths = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith('.png')])

def load_data(image_path, mask_path):
    # Đọc Ảnh
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0 
    
    # Đọc Mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.cast(mask > 0.5, tf.float32)
    return img, mask

# Băng chuyền Data (Đã update load cả cặp Ảnh-Mask)
dataset = tf.data.Dataset.from_tensor_slices((all_image_paths, all_mask_paths))
dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
print(f"✅ Đã tạo băng chuyền {len(all_image_paths)} cặp Ảnh-Mask chuẩn TF.")

# ======================================================================
# KIẾN TRÚC MẠNG BÊ NGUYÊN TỪ BẢN "CHẠY NGON" CỦA BRO
# ======================================================================
LATENT_DIM = 256 

class ConditionalVAE_Standard(tf.keras.Model):
    def __init__(self, latent_dim, **kwargs):
        super(ConditionalVAE_Standard, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        # --- 1. ENCODER (Vẫn y xì cũ) ---
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(256, 256, 6)),
            tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(256, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ])

        # --- 2. DECODER Z (Bung Z từ 1D ra 2D) ---
        self.decoder_z = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(16 * 16 * 256, activation="relu"),
            tf.keras.layers.Reshape((16, 16, 256)),
        ])
        
        # --- 3. DECODER UP (Kẹp 2D + Chống sọc + Không Ám màu) ---
        # Nhận khối kẹp (Z 256 kênh + Mask 3 kênh = 259 kênh)
        self.decoder_up = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(16, 16, 259)),
            
            tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            
            tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            
            tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            
            tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.Conv2D(3, 3, padding="same") # Logits thô
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
        # 1. Bung Z ra thành hình hộp 16x16
        z_spatial = self.decoder_z(z)
        # 2. Thu nhỏ Mask 256x256 về 16x16 (Tuyệt đối KHÔNG Flatten)
        mask_small = tf.image.resize(mask, [16, 16], method='nearest')
        # 3. Kẹp bánh mì 2 đứa nó lại với nhau theo trục kênh
        z_cond = tf.concat([z_spatial, mask_small], axis=-1)
        # 4. Bơm qua mạng UpSampling để vẽ
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

            log2pi = tf.cast(tf.math.log(2. * 3.14159265359), tf.float32)
            logpz = tf.reduce_sum(-0.5 * (z**2. + log2pi), axis=1)
            logqz_x = tf.reduce_sum(-0.5 * ((z - mean)**2. * tf.exp(-logvar) + logvar + log2pi), axis=1)

            kl_loss = -(logpz - logqz_x)
            loss = -tf.reduce_mean(logpx_z - BETA * kl_loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        clipped_gradients = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in gradients]
        
        self.optimizer.apply_gradients(zip(clipped_gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
# ======================================================================
# CHẠY VÀ LƯU FILE
# ======================================================================
model = ConditionalVAE_Standard(LATENT_DIM)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer)

# Chạy nháp để mồi Graph cho TensorFlow
print("⏳ Đang mồi TensorFlow Graph...")
dummy_img = tf.zeros((1, 256, 256, 3))
dummy_mask = tf.zeros((1, 256, 256, 3))
model([dummy_img, dummy_mask])

class BackupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            backup_path = os.path.join(SAVE_DIR, f"cvae_backup_epoch_{epoch+1}.weights.h5")
            self.model.save_weights(backup_path)
            print(f"\n💾 Đã lưu mốc an toàn tại Epoch {epoch+1}!")

csv_logger = tf.keras.callbacks.CSVLogger(CSV_PATH, separator=",", append=False)
checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
    filepath=BEST_WEIGHTS_PATH, monitor='loss', mode='min', save_best_only=True, save_weights_only=True, verbose=1
)

print("🚀 Khai hỏa Lò phản ứng CVAE (BẢN TENSORFLOW CHUẨN + CVAE)...")
model.fit(
    dataset, 
    epochs=2000, 
    callbacks=[BackupCallback(), csv_logger, checkpoint_best]
)

final_weights_path = os.path.join(SAVE_DIR, "cvae_weights_final.weights.h5")
model.save_weights(final_weights_path)
print("✅ Đã train xong!")