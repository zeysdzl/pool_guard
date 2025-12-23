import os
import cv2
from pathlib import Path

def crop_yolo_dataset(base_dir, split, output_base):
    """
    YOLO formatındaki veriyi crop yaparak classification formatına çevirir.
    """
    # Roboflow 'data.yaml' dosyasındaki sırayı buraya yaz. 
    # v6 sürümünde genellikle ['Adult', 'Child'] şeklindedir.
    classes = ['adult', 'child'] 
    
    img_dir = Path(base_dir) / split / 'images'
    label_dir = Path(base_dir) / split / 'labels'
    output_dir = Path(output_base) / split

    if not img_dir.exists():
        print(f"Hata: {img_dir} bulunamadı!")
        return

    print(f"İşleniyor: {split} seti...")

    for img_path in img_dir.glob("*.jpg"):
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w, _ = img.shape

        with open(label_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                parts = line.split()
                if not parts: continue
                
                cls_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                
                # Normalizasyonu piksele çevir
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)

                # Resmi sınırların dışına çıkmayacak şekilde kırp
                crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                
                if crop.size == 0: continue

                # Sınıf klasörünü oluştur ve kaydet
                class_name = classes[cls_id]
                save_path = output_dir / class_name
                save_path.mkdir(parents=True, exist_ok=True)
                
                file_name = f"{img_path.stem}_crop_{i}.jpg"
                cv2.imwrite(str(save_path / file_name), crop)

# --- AYARLAR ---
# Senin verdiğin yol:
RAW_DATA_PATH = r"C:\Users\zeyne\Desktop\pool_project\pool_guard_data\raw\Child-Adult Classification.v6i.yolov8"
# Çıktıların gideceği yer:
PROCESSED_PATH = r"C:\Users\zeyne\Desktop\pool_project\pool_guard_data\classifier"

# Train, Valid ve Test için çalıştır
for s in ['train', 'valid', 'test']:
    crop_yolo_dataset(RAW_DATA_PATH, s, PROCESSED_PATH)

print(f"\nİşlem tamam! Kırpılmış resimler burada: {PROCESSED_PATH}")