import os
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm # Để làm thanh tiến trình chạy cho đẹp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ======================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ======================================================================
MODEL_PATH = r"D:\BraTS2021_Split_3Mask\Models\attention_unet_wt_tc_et_resumed_best.keras"
TEST_IMG_DIR = r"D:\BraTS2021_Split_3Mask\val\images"
TEST_MASK_DIR = r"D:\BraTS2021_Split_3Mask\val\masks"

# ======================================================================
# 2. KHAI BÁO HÀM CUSTOM (ĐỂ LOAD MODEL)
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
# 3. CÁC HÀM TÍNH TOÁN METRICS CHO TỪNG KÊNH
# ======================================================================
def calculate_metrics(y_true, y_pred, smooth=1e-6):
    """Tính Dice, IoU, Precision, Recall cho 1 kênh (mảng numpy 1D)"""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    tp = np.sum(y_true * y_pred) # True Positive
    fp = np.sum((1 - y_true) * y_pred) # False Positive
    fn = np.sum(y_true * (1 - y_pred)) # False Negative

    dice = (2. * tp + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return dice, iou, precision, recall

# ======================================================================
# 4. LOAD MODEL VÀ DATA
# ======================================================================
print(f"🔄 Đang Load Model từ: {MODEL_PATH}...")
model = tf.keras.models.load_model(
    MODEL_PATH, 
    custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef}
)
print("✅ Load Model thành công!\n")

test_img_paths = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.png")))
test_mask_paths = sorted(glob.glob(os.path.join(TEST_MASK_DIR, "*.png")))

if not test_img_paths:
    print("❌ Lỗi: Không tìm thấy ảnh test.")
    exit()

print(f"🚀 Bắt đầu đánh giá trên {len(test_img_paths)} ảnh Test/Val...")

# Khởi tạo mảng lưu điểm cho 3 kênh [WT, TC, ET]
metrics_sum = {
    'WT': {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0},
    'TC': {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0},
    'ET': {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0}
}

valid_image_count = {'WT': 0, 'TC': 0, 'ET': 0}

# ======================================================================
# 5. CHẠY VÒNG LẶP ĐÁNH GIÁ (EVALUATION LOOP)
# ======================================================================
for idx in tqdm(range(len(test_img_paths)), desc="Đang chấm điểm"):
    # Đọc ảnh
    img = tf.io.read_file(test_img_paths[idx])
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img_input = tf.expand_dims(img, axis=0)

    # Đọc Mask
    mask_true = tf.io.read_file(test_mask_paths[idx])
    mask_true = tf.image.decode_png(mask_true, channels=3)
    mask_true = tf.cast(mask_true, tf.float32) / 255.0
    mask_true = tf.cast(mask_true > 0.5, tf.float32).numpy()

    # Dự đoán
    pred = model.predict(img_input, verbose=0)[0] 
    mask_pred = tf.cast(pred > 0.5, tf.float32).numpy()

    channels = ['WT', 'TC', 'ET']
    
    # Tính điểm từng kênh
    for c, channel_name in enumerate(channels):
        # Chỉ tính điểm nếu Ground Truth có chứa khối u đó (tránh nhiễu điểm số)
        if np.sum(mask_true[:, :, c]) > 0:
            dice, iou, precision, recall = calculate_metrics(mask_true[:, :, c], mask_pred[:, :, c])
            
            metrics_sum[channel_name]['dice'] += dice
            metrics_sum[channel_name]['iou'] += iou
            metrics_sum[channel_name]['precision'] += precision
            metrics_sum[channel_name]['recall'] += recall
            
            valid_image_count[channel_name] += 1

# ======================================================================
# 6. TÍNH TRUNG BÌNH VÀ IN BẢNG BÁO CÁO (Y CHANG ẢNH CỦA BRO)
# ======================================================================
print("\n" + "="*60)
print(f"{'Class':<8} {'Dice':<12} {'IoU':<12} {'Precision':<12} {'Recall':<12}")
print("-" * 60)

for channel_name in ['WT', 'TC', 'ET']:
    count = valid_image_count[channel_name]
    if count > 0:
        avg_dice = metrics_sum[channel_name]['dice'] / count
        avg_iou = metrics_sum[channel_name]['iou'] / count
        avg_precision = metrics_sum[channel_name]['precision'] / count
        avg_recall = metrics_sum[channel_name]['recall'] / count
        
        print(f"{channel_name:<8} {avg_dice:<12.6f} {avg_iou:<12.6f} {avg_precision:<12.6f} {avg_recall:<12.6f}")
    else:
        print(f"{channel_name:<8} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

print("="*60 + "\n")