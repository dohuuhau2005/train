import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt # Thư viện để show hình ảnh

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ======================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN 
# ======================================================================
MODEL_PATH = r"D:\BraTS2021_Split_3Mask_V2\Models\attention_unet_advanced_best.keras"
TEST_IMG_DIR = r"D:\BraTS2021_Split_3Mask_V2\val\images"
TEST_MASK_DIR = r"D:\BraTS2021_Split_3Mask_V2\val\masks"

# ======================================================================
# 2. KHAI BÁO HÀM CUSTOM (ĐỂ LOAD MODEL KHÔNG BỊ LỖI)
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
# 3. LOAD MODEL VÀ DATA
# ======================================================================
print(f"🔄 Đang Load Model từ: {MODEL_PATH}...")
model = tf.keras.models.load_model(
    MODEL_PATH, 
    custom_objects={
        'bce_dice_loss': bce_dice_loss, 
        'dice_coef': dice_coef,
        'dice_loss': dice_loss,
        'dice_per_channel': dice_per_channel
    }
)
print("✅ Load Model thành công!\n")

test_img_paths = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.png")))
test_mask_paths = sorted(glob.glob(os.path.join(TEST_MASK_DIR, "*.png")))

if not test_img_paths:
    print("❌ Lỗi: Không tìm thấy ảnh test.")
    exit()

# ======================================================================
# 4. CHỌN 1 ẢNH ĐỂ TEST VÀ SHOW LÊN MÀN HÌNH
# ======================================================================
# 🚨 CHÚ Ý: Đổi con số này để xem các lát cắt khác nhau (Ví dụ: 10, 50, 100...)
# Mẹo: Chọn mấy lát ở giữa (ví dụ 50-80) thì khối u thường to và rõ nhất
IMAGE_INDEX = 50 

img_path = test_img_paths[IMAGE_INDEX]
mask_path = test_mask_paths[IMAGE_INDEX]

print(f"🔍 Đang test trực quan trên ảnh số {IMAGE_INDEX}: {os.path.basename(img_path)}")

# --- Đọc và xử lý ảnh ---
img = tf.io.read_file(img_path)
img = tf.image.decode_png(img, channels=3)
img_tensor = tf.cast(img, tf.float32) / 255.0
img_input = tf.expand_dims(img_tensor, axis=0) # [1, 256, 256, 3] để đưa vào model

# --- Đọc và xử lý Mask chuẩn ---
mask_true = tf.io.read_file(mask_path)
mask_true = tf.image.decode_png(mask_true, channels=3)
mask_true_tensor = tf.cast(mask_true, tf.float32) / 255.0
mask_true_binary = tf.cast(mask_true_tensor > 0.5, tf.float32).numpy()

# --- Model thực hiện dự đoán ---
pred = model.predict(img_input)[0] 
mask_pred_binary = tf.cast(pred > 0.5, tf.float32).numpy()

# ======================================================================
# 5. VẼ LÊN MÀN HÌNH (SỬ DỤNG MATPLOTLIB)
# ======================================================================
plt.figure(figsize=(15, 5))

# HÌNH 1: ẢNH GỐC
plt.subplot(1, 3, 1)
plt.title("1. Ảnh Não Gốc (MRI)")
plt.imshow(img_tensor.numpy())
plt.axis('off')

# HÌNH 2: MASK CHUẨN
plt.subplot(1, 3, 2)
plt.title("2. Mask Chuẩn (Bác sĩ vẽ)")
plt.imshow(mask_true_binary)
plt.axis('off')

# HÌNH 3: MASK AI DỰ ĐOÁN
plt.subplot(1, 3, 3)
plt.title("3. Mask U-Net Dự Đoán")
plt.imshow(mask_pred_binary)
plt.axis('off')

plt.tight_layout()
plt.show()