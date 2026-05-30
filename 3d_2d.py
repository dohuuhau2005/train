import os
import cv2
import numpy as np
import nibabel as nib
from tqdm import tqdm

#Ép 3D thành 2D: Họ rút trích 3 chuẩn ảnh xịn nhất là T1Gd (t1ce), T2, và FLAIR rồi chập lại thành 1 tấm ảnh 3 kênh màu (như ảnh RGB bình thường) kích thước 256 x 256 x 3.Gộp nhãn (Binary Mask): File seg gốc có nhãn 1 (lõi), 2 (phù nề), 4 (bắt thuốc). Tác giả đã gom tất cả lại thành nhãn 1 (Khối u) và 0 (Nền đen)

# Cấu hình đường dẫn (Bro sửa lại cho đúng ổ đĩa máy bro nhé)
RAW_DATA_DIR = r"D:\BraTS2021\patients"

# 2 thư mục này code sẽ tự động tạo ra trên ổ D để chứa ảnh 2D, 
# bro cứ để nguyên hoặc đổi tên tùy thích.
OUTPUT_IMG_DIR = r"D:\BraTS2021_2D\images"
OUTPUT_MASK_DIR = r"D:\BraTS2021_2D\masks"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

def normalize_image(img):
    """Chuẩn hóa pixel ảnh về dải 0-255"""
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    return (img * 255).astype(np.uint8)

print("🚀 Bắt đầu máy thái thịt 3D sang 2D...")

patient_folders = os.listdir(RAW_DATA_DIR)

for patient_id in tqdm(patient_folders):
    patient_path = os.path.join(RAW_DATA_DIR, patient_id)
    if not os.path.isdir(patient_path): continue
    
    # Load 3 chuẩn MRI quan trọng nhất theo bài báo (T1ce, T2, FLAIR) và Mask
    try:
        t1ce = nib.load(os.path.join(patient_path, f"{patient_id}_t1ce.nii.gz")).get_fdata()
        t2   = nib.load(os.path.join(patient_path, f"{patient_id}_t2.nii.gz")).get_fdata()
        flair = nib.load(os.path.join(patient_path, f"{patient_id}_flair.nii.gz")).get_fdata()
        mask = nib.load(os.path.join(patient_path, f"{patient_id}_seg.nii.gz")).get_fdata()
    except Exception as e:
        continue # Bỏ qua nếu lỗi file
        
    # Kích thước gốc của BraTS thường là 240x240x155 (155 lát cắt)
    num_slices = flair.shape[2]
    
    for i in range(num_slices):
        # Chỉ lấy những lát cắt ở giữa (bỏ qua phần đỉnh đầu hoặc cằm vì toàn màu đen)
        # Đồng thời kiểm tra xem lát cắt đó có khối u không
        slice_mask = mask[:, :, i]
        if np.max(slice_mask) == 0: 
            continue # Lọc bỏ các lát cắt không có bệnh (Giảm tải cho máy)
            
        # 1. BIẾN ĐỔI MASK TỪ MULTI-CLASS SANG BINARY (0 và 1)
        slice_mask[slice_mask > 0] = 1 
        slice_mask = slice_mask.astype(np.uint8) * 255 # Nhân 255 để lưu ảnh trắng đen rõ ràng
        
        # 2. CHẬP 3 KÊNH MRI THÀNH ẢNH 3 MÀU (RGB giả lập)
        slice_t1ce = normalize_image(t1ce[:, :, i])
        slice_t2   = normalize_image(t2[:, :, i])
        slice_flair = normalize_image(flair[:, :, i])
        
        # Gộp lại: Kênh R=FLAIR, G=T2, B=T1ce
        combined_img = np.stack([slice_flair, slice_t2, slice_t1ce], axis=-1)
        
        # 3. RESIZE VỀ 256x256 NHƯ BÀI BÁO
        combined_img = cv2.resize(combined_img, (256, 256), interpolation=cv2.INTER_CUBIC)
        slice_mask = cv2.resize(slice_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Lưu ra ổ cứng
        img_name = f"{patient_id}_slice_{i}.png"
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, img_name), combined_img)
        cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, img_name), slice_mask)

print("✅ Xong! Kiểm tra ổ D xem thành quả nhé bro!")