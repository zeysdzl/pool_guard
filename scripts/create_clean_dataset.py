import cv2
import random
import shutil
import os
from pathlib import Path
from tqdm import tqdm

# --- AYARLAR ---
# Senin paylaştığın çalışan RAW veri yolu:
RAW_DATA_PATH = Path(r"C:\Users\zeyne\Desktop\pool_project\pool_guard_data\raw\Child-Adult Classification.v6i.yolov8")

# Yeni oluşturulacak temiz veri seti yolu:
OUTPUT_PATH = Path(r"C:\Users\zeyne\Desktop\pool_project\pool_guard_data\classifier_clean_500")

# Hedef Sayı (Her sınıf için)
TARGET_PER_CLASS = 250 

# Sınıflar (0: Adult, 1: Child) -> Senin veri setine göre sıralama
CLASSES = ["adult", "child"]

def create_structure():
    if OUTPUT_PATH.exists():
        try: shutil.rmtree(OUTPUT_PATH)
        except: pass
    
    for split in ["train", "valid", "test"]:
        for cls in CLASSES:
            (OUTPUT_PATH / split / cls).mkdir(parents=True, exist_ok=True)
    print("✅ Klasör yapısı temizlendi ve oluşturuldu.")

def collect_and_process():
    print(f"📂 Kaynak taranıyor: {RAW_DATA_PATH}")
    
    # Geçici hafıza (RAM'de tutacağız, sonra karıştırıp kaydedeceğiz)
    collected_crops = {"adult": [], "child": []}
    
    # Sadece 'train' klasörüne bakmak yeterli olur (en çok veri orada)
    # Eğer yetmezse valid'e de bakarız ama 20k veri var demiştin, yeter.
    split_dir = RAW_DATA_PATH / "train"
    img_dir = split_dir / "images"
    label_dir = split_dir / "labels"

    if not img_dir.exists():
        print(f"❌ HATA: {img_dir} bulunamadı!")
        return

    # Tüm resimleri listele
    image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    print(f"🔍 Toplam {len(image_files)} resim bulundu. İşleniyor...")

    # Karıştır ki hep videonun başındaki aynı kareler gelmesin
    random.shuffle(image_files)

    for img_path in tqdm(image_files):
        # Yeterli sayıya ulaştık mı?
        if len(collected_crops["adult"]) >= TARGET_PER_CLASS and \
           len(collected_crops["child"]) >= TARGET_PER_CLASS:
            break

        # Label dosyasını bul (isim.jpg -> isim.txt)
        label_path = label_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            continue

        # Resmi Oku
        img = cv2.imread(str(img_path))
        if img is None: continue
        h_img, w_img, _ = img.shape

        # Label Oku
        with label_path.open() as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            try:
                cls_id = int(parts[0]) # 0 veya 1
                cx, cy, w, h = map(float, parts[1:5])
            except: continue

            # Koordinat Hesapla
            x1 = int((cx - w / 2) * w_img)
            y1 = int((cy - h / 2) * h_img)
            x2 = int((cx + w / 2) * w_img)
            y2 = int((cy + h / 2) * h_img)

            # Sınırları düzelt
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            # KES (CROP)
            crop = img[y1:y2, x1:x2]

            # Çok küçükleri at (Çöp veri olmasın)
            if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
                continue

            # İlgili listeye ekle (limit dolmadıysa)
            class_name = CLASSES[cls_id] # 0->adult, 1->child
            
            if len(collected_crops[class_name]) < TARGET_PER_CLASS:
                collected_crops[class_name].append(crop)

    return collected_crops

def save_splits(crops_data):
    print("\n💾 Veriler diske yazılıyor...")
    
    # Dağılım Oranları: %70 Train, %20 Valid, %10 Test
    split_ratios = {"train": 0.7, "valid": 0.2, "test": 0.1}

    for cls_name, crop_list in crops_data.items():
        total = len(crop_list)
        print(f"   🔹 {cls_name.upper()}: {total} adet toplandı.")
        
        if total == 0:
            print(f"   ⚠️ UYARI: {cls_name} için hiç veri bulunamadı!")
            continue

        # Son bir karıştırma
        random.shuffle(crop_list)

        n_train = int(total * split_ratios["train"])
        n_valid = int(total * split_ratios["valid"])
        
        splits = {
            "train": crop_list[:n_train],
            "valid": crop_list[n_train:n_train+n_valid],
            "test": crop_list[n_train+n_valid:]
        }

        for split_name, images in splits.items():
            save_dir = OUTPUT_PATH / split_name / cls_name
            for i, img in enumerate(images):
                filename = f"{cls_name}_{i:04d}.jpg"
                cv2.imwrite(str(save_dir / filename), img)

    print(f"\n✅ İŞLEM TAMAM! Yeni temiz veri seti: {OUTPUT_PATH}")

if __name__ == "__main__":
    create_structure()
    data = collect_and_process()
    save_splits(data)