import os
import tensorflow as tf

# 🚨 GIỮ NGUYÊN TẮT MIXED PRECISION NHƯ BẢN GỐC
BATCH_SIZE = 16

# ======================================================================
# 1. BĂNG CHUYỀN LOAD ẢNH
# ======================================================================
IMG_DIR = r"D:\BraTS2021_Split_3Mask\train\images" 
all_image_paths = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith('.png')]

def load_img(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0 
    return img

dataset = tf.data.Dataset.from_tensor_slices(all_image_paths)
dataset = dataset.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
print(f"✅ Đã tạo băng chuyền {len(all_image_paths)} ảnh chuẩn TF.")

# ======================================================================
# 2. KIẾN TRÚC MẠNG BÊ NGUYÊN TỪ BẢN MỚI NHẤT (CÓ THÊM HÀM CALL)
# ======================================================================
LATENT_DIM = 256 

class CVAE_TF_Standard(tf.keras.Model):
    def __init__(self, latent_dim, **kwargs):
        super(CVAE_TF_Standard, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

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
            tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding="same"),
        ])

    # 🚨 HÀM BẮT BUỘC ĐỂ KHÔNG BỊ LỖI KHI MỒI DUMMY_INPUT
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

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            mean, logvar = self.encode(data)
            z = self.reparameterize(mean, logvar)
            x_logit = self.decode(z)

            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=data)
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

            log2pi = tf.cast(tf.math.log(2. * 3.14159265359), tf.float32)
            logpz = tf.reduce_sum(-0.5 * (z**2. + log2pi), axis=1)
            logqz_x = tf.reduce_sum(-0.5 * ((z - mean)**2. * tf.exp(-logvar) + logvar + log2pi), axis=1)

            loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)

        gradients = tape.gradient(loss, self.trainable_variables)
        clipped_gradients = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in gradients]
        
        self.optimizer.apply_gradients(zip(clipped_gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        
        return {"loss": self.loss_tracker.result()}

# BÙA HỘ MỆNH: Lưu mỗi 5 Epoch cho an toàn
class BackupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            self.model.save_weights(rf"D:\BraTS2021_2D\cvae_backup_epoch_{epoch+1}.h5")
            print(f"\n💾 Đã lưu mốc an toàn tại Epoch {epoch+1}!")

backup_epochs = BackupCallback()

# ======================================================================
# 3. GIAI ĐOẠN "HỒI SINH" VÀ TRAIN TIẾP
# ======================================================================
# 🔴 BRO SỬA 3 THÔNG SỐ NÀY ĐỂ CHẠY TIẾP TỪ EPOCH 4 HOẶC 5 NHÉ:
CHECKPOINT_PATH = r"D:\BraTS2021_2D\cvae_backup_epoch_4.h5"  # Trỏ vào file Epoch 4 bro vừa khoe
CURRENT_EPOCH = 4                                               # Mốc bắt đầu chạy tiếp
TARGET_EPOCH = 3000                                             # Đích đến
# ======================================================================

model = CVAE_TF_Standard(LATENT_DIM)

# Mồi 1 nhịp dummy_input để dựng khung xương trước khi nạp linh hồn
dummy_input = tf.zeros((1, 256, 256, 3))
model(dummy_input)

# Nạp Weights
model.load_weights(CHECKPOINT_PATH)
print(f"🔥 Nạp thành công nội công từ mốc Epoch {CURRENT_EPOCH}!")

# 💡 Mẹo: Giảm Learning Rate xuống 0.00005 để AI điêu khắc chi tiết mượt hơn
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
model.compile(optimizer=optimizer)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss', 
    patience=50, 
    restore_best_weights=True
)

print(f"🚀 Tiếp tục hành trình tu tiên đến cảnh giới Epoch {TARGET_EPOCH}...")

# Chạy fit với initial_epoch
history = model.fit(
    dataset, 
    epochs=TARGET_EPOCH, 
    initial_epoch=CURRENT_EPOCH, 
    callbacks=[early_stopping, backup_epochs]
)

# Lưu mốc cuối
model.save_weights(r"D:\BraTS2021_2D\cvae_weights_resumed_final.h5")
print(f"✅ Đã train xong giai đoạn mở rộng!")