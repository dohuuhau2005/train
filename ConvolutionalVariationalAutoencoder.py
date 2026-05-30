import os
import tensorflow as tf

# 🚨 TẮT HOÀN TOÀN MIXED PRECISION (Kẻ thù của VAE)
# Giảm Batch Size xuống 8 để con RTX 4060 không bị tràn VRAM
BATCH_SIZE = 16

IMG_DIR = r"D:\BraTS2021_Split_3Mask\train\images" 
all_image_paths = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith('.png')]

def load_img(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0 
    return img

# Băng chuyền Data
dataset = tf.data.Dataset.from_tensor_slices(all_image_paths)
dataset = dataset.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
print(f"✅ Đã tạo băng chuyền {len(all_image_paths)} ảnh chuẩn TF.")

# ======================================================================
# KIẾN TRÚC MẠNG BÊ NGUYÊN TỪ TENSORFLOW TUTORIAL 
# ======================================================================
LATENT_DIM = 256 # Tăng nhẹ latent để não không bị mờ

class CVAE_TF_Standard(tf.keras.Model):
    def __init__(self, latent_dim, **kwargs):
        super(CVAE_TF_Standard, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        # --- 1. ENCODER ---
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(256, 256, 3)),
            tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(256, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Flatten(),
            # Cắt đôi output ra z_mean và z_log_var như trong tutorial
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ])

        # --- 2. DECODER ---
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(16 * 16 * 256, activation="relu"),
            tf.keras.layers.Reshape((16, 16, 256)),
            tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu"),
            # 🚨 QUAN TRỌNG: KHÔNG CÓ SIGMOID Ở ĐÂY (Xuất Logits thô)
            tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding="same"),
        ])

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        # 🛡️ CHỐNG NỔ SỐ KHI DÙNG HÀM EXP()
        logvar = tf.clip_by_value(logvar, -10.0, 10.0)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z):
        return self.decoder(z)

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            mean, logvar = self.encode(data)
            z = self.reparameterize(mean, logvar)
            x_logit = self.decode(z)

            # 1. CROSS ENTROPY LOSS (Chuẩn công thức TF Tutorial)
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=data)
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

            # 2. KL DIVERGENCE (Chuẩn Log Normal PDF của TF)
            log2pi = tf.cast(tf.math.log(2. * 3.14159265359), tf.float32)
            logpz = tf.reduce_sum(-0.5 * (z**2. + log2pi), axis=1)
            logqz_x = tf.reduce_sum(-0.5 * ((z - mean)**2. * tf.exp(-logvar) + logvar + log2pi), axis=1)

            # Tổng cộng (Maximize Evidence Lower Bound)
            loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)

        gradients = tape.gradient(loss, self.trainable_variables)
        
        # 🛡️ CẮT GRADIENT: Bắt buộc để tránh model bị sốc nhiệt và học ra màu xám
        clipped_gradients = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in gradients]
        
        self.optimizer.apply_gradients(zip(clipped_gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        
        return {"loss": self.loss_tracker.result()}

# ======================================================================
# LƯU FILE THEO CHUẨN MỚI
# ======================================================================
model = CVAE_TF_Standard(LATENT_DIM)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer)

class BackupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 2 == 0:
            # Lưu full model thay vì chỉ weights cho chắc ăn
            self.model.save_weights(rf"D:\BraTS2021_2D\cvae_backup_epoch_{epoch+1}.h5")
            print(f"\n💾 Đã lưu mốc an toàn tại Epoch {epoch+1}!")

print("🚀 Khai hỏa Lò phản ứng CVAE (BẢN TENSORFLOW TUTORIAL 100%)...")
model.fit(dataset, epochs=2000, callbacks=[BackupCallback()])

model.save_weights(r"D:\BraTS2021_2D\cvae_weights_final.h5")
print("✅ Đã train xong!")