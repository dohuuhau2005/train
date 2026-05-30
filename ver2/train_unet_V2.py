import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import albumentations as A
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision

# ======================================================================
# BAT MIXED PRECISION DE TANG TOC VA TIET KIEM VRAM
# ======================================================================
mixed_precision.set_global_policy('mixed_float16')
print("Da bat Mixed Precision. Card RTX 4060 se chay het toc luc!")

# ======================================================================
# 1. CAU HINH DUONG DAN DU LIEU (TRO VAO THU MUC ADVANCED MOI)
# ======================================================================
TRAIN_IMG_REAL = r"D:\BraTS2021_Split_3Mask_V2\train\images"
TRAIN_MASK_REAL = r"D:\BraTS2021_Split_3Mask_V2\train\masks"

# NEU BRO DA SINH ANH CVAE, HAY TRO VAO DAY, NEU CHUA THI DE TRONG CUNG DUOC
TRAIN_IMG_FAKE = r"" 
TRAIN_MASK_FAKE = r""

VAL_IMG_DIR = r"D:\BraTS2021_Split_3Mask_V2\val\images"
VAL_MASK_DIR = r"D:\BraTS2021_Split_3Mask_V2\val\masks"

CKPT_DIR = r"D:\BraTS2021_Split_3Mask_V2\Models"
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(CKPT_DIR, "attention_unet_advanced_best.keras")
CSV_PATH = os.path.join(CKPT_DIR, "advanced_training_history.csv")

# ======================================================================
# 2. XAY DUNG BANG CHUYEN DATA + ALBUMENTATIONS TANG CUONG
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
    aug_img, aug_mask = tf.py_function(func=albumentations_fn, 
                                       inp=[img, mask], 
                                       Tout=[tf.float32, tf.float32])
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

print(f"Train batches: {len(train_ds)}")
print(f"Val batches  : {len(val_ds)}")

# ======================================================================
# 3. MO HINH ATTENTION U-NET 
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

    outputs = layers.Conv2D(n_classes, 1, activation="sigmoid", dtype='float32')(c8)
    return models.Model(inputs, outputs)

# ======================================================================
# 4. HAM LOSS VA METRICS DA KENH 
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
# 5. KHOI TAO VA BAT DAU HUAN LUYEN
# ======================================================================
model = build_attention_unet(input_shape=(256,256,3), n_classes=3)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=bce_dice_loss,
    metrics=[dice_coef]
)

class BackupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            backup_path = os.path.join(CKPT_DIR, f"advanced_backup_epoch_{epoch+1}.keras")
            self.model.save(backup_path)
            print(f"\n[AUTO BACKUP] Da luu moc an toan tai Epoch {epoch+1}!")

backup_10_epochs = BackupCallback()

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
        patience=15, 
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
    tf.keras.callbacks.CSVLogger(
        CSV_PATH,
        separator=",",
        append=False
    ),
    backup_10_epochs 
]

print("Khai hoa qua trinh Train Attention U-Net (Full Augmentation)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100, 
    callbacks=callbacks
)

print(f"Training hoan tat! Model va file CSV luu tai: {CKPT_DIR}")