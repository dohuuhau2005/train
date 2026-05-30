import os
import cv2
import numpy as np
import nibabel as nib
from tqdm import tqdm

# ======================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ======================================================================
RAW_DATA_DIR = r"D:\BraTS2021\patients"
OUTPUT_IMG_DIR = r"D:\BraTS2021_2D_3Mask_V2\images"
OUTPUT_MASK_DIR = r"D:\BraTS2021_2D_3Mask_V2\masks"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# ======================================================================
# CÁC HÀM TIỀN XỬ LÝ (ĐÃ FIX LỖI NỀN MÀU & LỖI ZOOM)
# ======================================================================

def normalize_z_score(img):
    """Chuẩn hóa Z-score tuyệt đối chỉ dành cho vùng NÃO, ép nền thành đen (0)"""
    mask = img > 0
    if np.sum(mask) > 0:
        # 1. Trích xuất chỉ các pixel thuộc về não
        brain_pixels = img[mask]
        mean = brain_pixels.mean()
        std = brain_pixels.std()
        
        # 2. Tính Z-score cho riêng não
        img_norm = np.zeros_like(img, dtype=np.float32)
        img_norm[mask] = (img[mask] - mean) / (std + 1e-8)
        
        # 3. Scale riêng phần não về dải 0-255 để lưu ảnh
        min_val = np.min(img_norm[mask])
        max_val = np.max(img_norm[mask])
        
        img_final = np.zeros_like(img, dtype=np.uint8)
        img_final[mask] = ((img_norm[mask] - min_val) / (max_val - min_val + 1e-8) * 255).astype(np.uint8)
        
        return img_final
    return np.zeros_like(img, dtype=np.uint8)

def crop_roi_2d(image, mask, padding=10):
    """Cắt vùng quan tâm (ROI) dựa trên TOÀN BỘ SỌ NÃO thay vì khối u"""
    # Tìm pixel thuộc não (bất kỳ kênh MRI nào sáng > 0)
    brain_mask = np.sum(image, axis=-1) > 0
    coords = np.where(brain_mask)
    
    if len(coords[0]) == 0:
        return image, mask

    # Bounding box ôm sát sọ não
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()

    # Mở rộng vùng cắt thêm một khoảng padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image.shape[0], x_max + padding)
    y_max = min(image.shape[1], y_max + padding)

    # Cắt ảnh và mask theo bounding box của NÃO
    cropped_img = image[x_min:x_max, y_min:y_max, :]
    cropped_mask = mask[x_min:x_max, y_min:y_max, :]
    
    return cropped_img, cropped_mask

# ======================================================================
# CHẠY MÁY THÁI THỊT
# ======================================================================
print("🚀 Bắt đầu thái thịt 3D sang 2D (Đã fix lỗi Crop và nền đen)...")
patient_folders = os.listdir(RAW_DATA_DIR)

for patient_id in tqdm(patient_folders):
    patient_path = os.path.join(RAW_DATA_DIR, patient_id)
    if not os.path.isdir(patient_path): continue
    
    try:
        t1ce = nib.load(os.path.join(patient_path, f"{patient_id}_t1ce.nii.gz")).get_fdata()
        t2   = nib.load(os.path.join(patient_path, f"{patient_id}_t2.nii.gz")).get_fdata()
        flair = nib.load(os.path.join(patient_path, f"{patient_id}_flair.nii.gz")).get_fdata()
        mask = nib.load(os.path.join(patient_path, f"{patient_id}_seg.nii.gz")).get_fdata()
    except Exception as e:
        continue 
        
    num_slices = flair.shape[2]
    
    for i in range(num_slices):
        slice_mask_raw = mask[:, :, i]
        
        # BỘ LỌC RÁC: Vứt bỏ các lát cắt có quá ít điểm ảnh khối u (dưới 50 pixel)
        if np.sum(slice_mask_raw > 0) < 50: 
            continue 
            
        wt = np.zeros_like(slice_mask_raw)
        tc = np.zeros_like(slice_mask_raw)
        et = np.zeros_like(slice_mask_raw)

        wt[(slice_mask_raw == 1) | (slice_mask_raw == 2) | (slice_mask_raw == 4)] = 1
        tc[(slice_mask_raw == 1) | (slice_mask_raw == 4)] = 1
        et[slice_mask_raw == 4] = 1

        mask_rgb = np.stack([wt, tc, et], axis=-1).astype(np.uint8) * 255
        
        # Chuẩn hóa Z-score từng xung
        slice_t1ce = normalize_z_score(t1ce[:, :, i])
        slice_t2   = normalize_z_score(t2[:, :, i])
        slice_flair = normalize_z_score(flair[:, :, i])
        
        # Chập 3 xung lại (Ảnh sẽ có màu vì độ sáng 3 xung khác nhau -> BÌNH THƯỜNG)
        combined_img = np.stack([slice_flair, slice_t2, slice_t1ce], axis=-1)
        
        # CẮT ROI (Ôm sát sọ não)
        combined_img, mask_rgb = crop_roi_2d(combined_img, mask_rgb)
        
        # Resize về 256x256
        combined_img = cv2.resize(combined_img, (256, 256), interpolation=cv2.INTER_CUBIC)
        mask_rgb = cv2.resize(mask_rgb, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        img_name = f"{patient_id}_slice_{i}.png"
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, img_name), combined_img)
        cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, img_name), mask_bgr)

print("✅ Hoàn tất! Tập dữ liệu chuẩn y khoa đã sẵn sàng (Nền đen tuyệt đối, nhìn rõ cả sọ não)!")