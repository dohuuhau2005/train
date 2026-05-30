import os
import glob
import numpy as np
from tqdm import tqdm

# Ép TensorFlow câm mồm bớt mấy cái log Warning loằng ngoằng
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision

# Bật Mixed Precision cho đồng bộ với file Train
mixed_precision.set_global_policy('mixed_float16')
print("⚡ Đã bật Mixed Precision cho khâu Test. Đọc nhanh như chớp!")

# ======================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU LOCAL
# ======================================================================
# 🚨 CHÚ Ý: Đảm bảo file weights ông train lúc nãy đuôi là .h5 nhé!
MODEL_WEIGHTS_PATH = r"D:\BraTS2021_Split_3Mask_V2\Models\attention_unet_advanced_best.h5"

TEST_IMG_DIR = r"D:\BraTS2021_Split_3Mask_V2\val\images"
TEST_MASK_DIR = r"D:\BraTS2021_Split_3Mask_V2\val\masks"

# ======================================================================
# 2. XÂY DỰNG LẠI CÁI VỎ MẠNG (ATTENTION U-NET)
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

    # Lớp output trả về float32 để chống sai số của mixed_precision
    outputs = layers.Conv2D(n_classes, 1, activation="sigmoid", dtype='float32')(c8)
    return models.Model(inputs, outputs)

# ======================================================================
# 3. HÀM TÍNH TOÁN METRICS Y KHOA CHUẨN XÁC
# ======================================================================
def calculate_metrics(y_true, y_pred, smooth=1e-6):
    """Tính Dice, IoU, Precision, Recall cho từng kênh"""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    tp = np.sum(y_true * y_pred) # True Positive: U thật, đoán trúng U
    fp = np.sum((1 - y_true) * y_pred) # False Positive: Não khỏe, đoán nhầm là U
    fn = np.sum(y_true * (1 - y_pred)) # False Negative: U thật, nhưng bị bỏ sót

    dice = (2. * tp + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return dice, iou, precision, recall

# ======================================================================
# 4. KHỞI TẠO VỎ MẠNG VÀ "NHẬP HỒN" (LOAD WEIGHTS)
# ======================================================================
print("\n⏳ Đang đúc khuôn kiến trúc Attention U-Net...")
model = build_attention_unet() 

# Mồi Graph bằng 1 cái ảnh ảo (Trắng tinh) để TensorFlow khởi tạo RAM
dummy_img = tf.zeros((1, 256, 256, 3))
model(dummy_img)

if not os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"\n❌ LỖI NGHIÊM TRỌNG: Không tìm thấy file weights tại {MODEL_WEIGHTS_PATH}")
    print("👉 Ông check lại xem đường dẫn đúng chưa, và file đã có mặt trong máy chưa nha!")
    exit()

print(f"🔄 Đang nhập hồn từ file weights: {MODEL_WEIGHTS_PATH}...")
model.load_weights(MODEL_WEIGHTS_PATH)
print("✅ Load Model thành công tưng bừng!\n")

# ======================================================================
# 5. ĐỌC ẢNH VÀ CHUẨN BỊ VÒNG LẶP TEST
# ======================================================================
test_img_paths = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.png")))
test_mask_paths = sorted(glob.glob(os.path.join(TEST_MASK_DIR, "*.png")))

if not test_img_paths or len(test_img_paths) != len(test_mask_paths):
    print("❌ Lỗi: Số lượng ảnh và Mask không khớp hoặc không tìm thấy ảnh. Kiểm tra lại đường dẫn!")
    exit()

print(f"🚀 Bắt đầu đánh giá Model trên {len(test_img_paths)} lát cắt (slices)...")

metrics_sum = {
    'WT': {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0},
    'TC': {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0},
    'ET': {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0}
}
valid_image_count = {'WT': 0, 'TC': 0, 'ET': 0}

# ======================================================================
# 6. VÒNG LẶP CHẤM ĐIỂM (CHỈ CHẤM NHỮNG LÁT CẮT CÓ U)
# ======================================================================
for idx in tqdm(range(len(test_img_paths)), desc="Đang quét ảnh MRI"):
    # Đọc & Tiền xử lý Ảnh (Scale 0-1)
    img = tf.io.read_file(test_img_paths[idx])
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img_input = tf.expand_dims(img, axis=0) # Đẩy thành tensor (1, 256, 256, 3)

    # Đọc & Tiền xử lý Mask
    mask_true = tf.io.read_file(test_mask_paths[idx])
    mask_true = tf.image.decode_png(mask_true, channels=3)
    mask_true = tf.cast(mask_true, tf.float32) / 255.0
    mask_true = tf.cast(mask_true > 0.5, tf.float32).numpy()

    # Nhờ con AI phán (Trả về xác suất 0->1)
    pred = model.predict(img_input, verbose=0)[0] 
    # Ép kiểu thành 0 và 1 cứng (Hard threshold)
    mask_pred = tf.cast(pred > 0.5, tf.float32).numpy()

    channels = ['WT', 'TC', 'ET']
    
    # Duyệt qua từng loại U (Từng màu của Mask)
    for c, channel_name in enumerate(channels):
        # 🛡️ LUẬT Y KHOA: Chỉ tính điểm lát cắt này NẾU nó thật sự chứa khối u (mask_true > 0)
        if np.sum(mask_true[:, :, c]) > 0:
            dice, iou, precision, recall = calculate_metrics(mask_true[:, :, c], mask_pred[:, :, c])
            
            metrics_sum[channel_name]['dice'] += dice
            metrics_sum[channel_name]['iou'] += iou
            metrics_sum[channel_name]['precision'] += precision
            metrics_sum[channel_name]['recall'] += recall
            
            valid_image_count[channel_name] += 1

# ======================================================================
# 7. TỔNG KẾT VÀ IN BÁO CÁO MANG ĐI BẢO VỆ ĐỒ ÁN
# ======================================================================
print("\n" + "="*70)
print(f"{'Loại U (Class)':<15} | {'Dice Score':<10} | {'IoU':<10} | {'Precision':<10} | {'Recall':<10}")
print("-" * 70)

for channel_name in ['WT', 'TC', 'ET']:
    count = valid_image_count[channel_name]
    if count > 0:
        avg_dice = metrics_sum[channel_name]['dice'] / count
        avg_iou = metrics_sum[channel_name]['iou'] / count
        avg_precision = metrics_sum[channel_name]['precision'] / count
        avg_recall = metrics_sum[channel_name]['recall'] / count
        
        # In các thông số với 4 chữ số thập phân cho đẹp mắt
        print(f"{channel_name:<15} | {avg_dice:<10.4f} | {avg_iou:<10.4f} | {avg_precision:<10.4f} | {avg_recall:<10.4f}")
    else:
        print(f"{channel_name:<15} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")

print("="*70)
print(f"📌 Tổng số lát cắt chứa u (WT): {valid_image_count['WT']} ảnh")
print(f"📌 Tổng số lát cắt chứa u (TC): {valid_image_count['TC']} ảnh")
print(f"📌 Tổng số lát cắt chứa u (ET): {valid_image_count['ET']} ảnh")
print("🎉 Đánh giá hoàn tất! Chúc bro lấy điểm A+ đồ án nhé!")