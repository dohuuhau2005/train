import os
import cv2
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Kênh R (Đỏ): Là vùng WT (Whole Tumor - Toàn bộ u).

# Kênh G (Xanh lá): Là vùng TC (Tumor Core - Lõi u).

# Kênh B (Xanh dương): Là vùng ET (Enhancing Tumor - U bắt thuốc).
# Cấu hình đường dẫn (Bro sửa lại cho đúng ổ đĩa máy bro nhé)
RAW_DATA_DIR = r"D:\BraTS2021\patients"

OUTPUT_IMG_DIR = r"D:\BraTS2021_2D_3Mask\images"
OUTPUT_MASK_DIR = r"D:\BraTS2021_2D_3Mask\masks"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

def normalize_image(img):
    """Chuẩn hóa pixel ảnh về dải 0-255"""
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    return (img * 255).astype(np.uint8)

print("🚀 Bắt đầu máy thái thịt 3D sang 2D (Phiên bản 3 Mask WT/TC/ET)...")

patient_folders = os.listdir(RAW_DATA_DIR)

for patient_id in tqdm(patient_folders):
    patient_path = os.path.join(RAW_DATA_DIR, patient_id)
    if not os.path.isdir(patient_path): continue
    
    # Load 3 chuẩn MRI và Mask
    try:
        t1ce = nib.load(os.path.join(patient_path, f"{patient_id}_t1ce.nii.gz")).get_fdata()
        t2   = nib.load(os.path.join(patient_path, f"{patient_id}_t2.nii.gz")).get_fdata()
        flair = nib.load(os.path.join(patient_path, f"{patient_id}_flair.nii.gz")).get_fdata()
        mask = nib.load(os.path.join(patient_path, f"{patient_id}_seg.nii.gz")).get_fdata()
    except Exception as e:
        continue # Bỏ qua nếu lỗi file
        
    num_slices = flair.shape[2]
    
    for i in range(num_slices):
        slice_mask_raw = mask[:, :, i]
        if np.max(slice_mask_raw) == 0: 
            continue # Lọc bỏ các lát cắt không có bệnh
            
        # ==========================================================
        # 1. BIẾN ĐỔI MASK TỪ GỐC SANG 3 VÙNG (WT, TC, ET)
        # ==========================================================
        wt = np.zeros_like(slice_mask_raw)
        tc = np.zeros_like(slice_mask_raw)
        et = np.zeros_like(slice_mask_raw)

        # Nhãn gốc BraTS: 1 (NCR/NET), 2 (Edema), 4 (ET)
        
        # WT (Whole Tumor): Bao gồm tất cả (1, 2, 4)
        wt[(slice_mask_raw == 1) | (slice_mask_raw == 2) | (slice_mask_raw == 4)] = 1
        
        # TC (Tumor Core): Bao gồm lõi hoại tử và phần bắt thuốc (1, 4)
        tc[(slice_mask_raw == 1) | (slice_mask_raw == 4)] = 1
        
        # ET (Enhancing Tumor): Chỉ bao gồm phần bắt thuốc (4)
        et[slice_mask_raw == 4] = 1

        # Gộp lại thành ảnh 3 kênh (Kênh 0: WT, Kênh 1: TC, Kênh 2: ET)
        # Nhân 255 để lưu file PNG không bị đen thui
        mask_rgb = np.stack([wt, tc, et], axis=-1).astype(np.uint8) * 255
        
        # ==========================================================
        # 2. CHẬP 3 KÊNH MRI THÀNH ẢNH 3 MÀU (RGB giả lập)
        # ==========================================================
        slice_t1ce = normalize_image(t1ce[:, :, i])
        slice_t2   = normalize_image(t2[:, :, i])
        slice_flair = normalize_image(flair[:, :, i])
        combined_img = np.stack([slice_flair, slice_t2, slice_t1ce], axis=-1)
        
        # ==========================================================
        # 3. RESIZE VỀ 256x256 NHƯ BÀI BÁO
        # ==========================================================
        combined_img = cv2.resize(combined_img, (256, 256), interpolation=cv2.INTER_CUBIC)
        # BẮT BUỘC dùng INTER_NEAREST cho mask để không bị nhòe ranh giới 0 và 255
        mask_rgb = cv2.resize(mask_rgb, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # ==========================================================
        # 4. LƯU RA Ổ CỨNG
        # ==========================================================
        # Vì OpenCV tự động lưu ảnh theo hệ BGR, ta phải dịch ngược RGB -> BGR 
        # Để sau này lúc load bằng TensorFlow (nó đọc RGB), 3 kênh WT, TC, ET không bị lộn xộn
        mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)

        img_name = f"{patient_id}_slice_{i}.png"
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, img_name), combined_img)
        cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, img_name), mask_bgr)

print("✅ Thái thịt thành công! Tập dữ liệu 3 Mask đã sẵn sàng!")