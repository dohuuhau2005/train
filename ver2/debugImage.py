import os

# Trỏ vào thư mục Split 3 Mask mà bro vừa tạo xong
BASE_DIR = r"D:\BraTS2021_Split_3Mask_V2"

splits = ['train', 'val', 'test']
total_images = 0
total_masks = 0

print(f"📊 BÁO CÁO KIỂM KÊ KHO DỮ LIỆU: {BASE_DIR}")
print("=" * 55)

for split in splits:
    img_dir = os.path.join(BASE_DIR, split, 'images')
    mask_dir = os.path.join(BASE_DIR, split, 'masks')
    
    # Kiểm tra xem thư mục có tồn tại không trước khi đếm
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f"⚠️ LỖI: Không tìm thấy thư mục {split.upper()}! Bro check lại đường dẫn nhé.")
        continue
        
    # Đếm số lượng file trong thư mục
    img_count = len(os.listdir(img_dir))
    mask_count = len(os.listdir(mask_dir))
    
    print(f"📁 Tập {split.upper()}:")
    print(f"   🖼️  Images: {img_count:,} file")
    print(f"   🎭  Masks:  {mask_count:,} file")
    print("-" * 30)
    
    total_images += img_count
    total_masks += mask_count

print("=" * 55)
print(f"📈 TỔNG KẾT TOÀN BỘ KHO:")
print(f"   -> Tổng số Images: {total_images:,} file")
print(f"   -> Tổng số Masks:  {total_masks:,} file")

# Kiểm tra chéo với con số gốc
if total_images == 81437 and total_masks == 81437:
    print("\n✅ HOÀN HẢO! CHUẨN XÁC 100%! Không rớt một byte nào! Bắt đầu rặn đẻ thôi bro! 🚀")
else:
    print(f"\n⚠️ CẢNH BÁO: Số lượng đang bị lệch so với 81.437. Bro cần check lại bước Copy!")