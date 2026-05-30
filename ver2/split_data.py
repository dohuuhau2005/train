import os
import random
import shutil
from tqdm import tqdm

# ======================================================================
# 1. TRỎ VÀO THƯ MỤC BẢN NÂNG CẤP (ĐÃ Z-SCORE & CROP ROI)
# ======================================================================
IMG_DIR = r"D:\BraTS2021_2D_3Mask_V2\images"
MASK_DIR = r"D:\BraTS2021_2D_3Mask_V2\masks"

# ======================================================================
# 2. TẠO THƯ MỤC SPLIT MỚI ĐỂ KHÔNG ĐỤNG CHẠM DATA CŨ
# ======================================================================
BASE_DIR = r"D:\BraTS2021_Split_3Mask_V2"

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(BASE_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, split, 'masks'), exist_ok=True)

all_images = os.listdir(IMG_DIR)
random.seed(42) # Chốt seed để nếu có chạy lại thì nó vẫn chia y như cũ
random.shuffle(all_images)

total = len(all_images)
train_end = int(total * 0.70)
val_end = train_end + int(total * 0.15)

train_files = all_images[:train_end]
val_files = all_images[train_end:val_end]
test_files = all_images[val_end:]

print(f"📊 Tổng số ảnh: {total}")
print(f"   -> Train (70%): {len(train_files)} ảnh")
print(f"   -> Validation (15%): {len(val_files)} ảnh")
print(f"   -> Test (15%): {len(test_files)} ảnh")

def move_files(files, split_name):
    print(f"\n🚀 Đang copy vào tập {split_name.upper()}...")
    for f in tqdm(files):
        shutil.copy(os.path.join(IMG_DIR, f), 
                    os.path.join(BASE_DIR, split_name, 'images', f))
        shutil.copy(os.path.join(MASK_DIR, f), 
                    os.path.join(BASE_DIR, split_name, 'masks', f))

move_files(train_files, 'train')
move_files(val_files, 'val')
move_files(test_files, 'test')

print("\n✅ CHIA 70-15-15 THÀNH CÔNG! SẴN SÀNG TRAIN!")