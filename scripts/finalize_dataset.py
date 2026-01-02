import os
import shutil
import random
from pathlib import Path

# --- AYARLAR ---
# Mevcut parçalı verilerinin olduğu ana klasör
SOURCE_DIR = Path(r"C:\Users\zeyne\Desktop\pool_project\pool_guard_data\classifier_clean_500")

# Yeni organize edilmiş veri setinin kurulacağı yer
FINAL_DIR = Path(r"C:\Users\zeyne\Desktop\pool_project\pool_guard_data\classifier_final")

# 10-20-70 Kuralı
RATIOS = {"train": 0.7, "valid": 0.2, "test": 0.1}

def setup_folders():
    if FINAL_DIR.exists():
        shutil.rmtree(FINAL_DIR)
    for split in RATIOS.keys():
        for cls in ["adult", "child"]:
            (FINAL_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    print(f"✅ Hedef klasör yapısı hazırlandı: {FINAL_DIR}")

def organize():
    for cls in ["adult", "child"]:
        # 1. Tüm alt klasörlerdeki (train, test, valid) o sınıfa ait resimleri topla
        # Hiçbir resmi silmiyoruz, hepsini bir listeye alıyoruz
        all_images = list(SOURCE_DIR.rglob(f"{cls}/*.*"))
        random.shuffle(all_images)
        
        total_count = len(all_images)
        print(f"📦 {cls.upper()}: Toplam {total_count} resim bulundu. Dağıtılıyor...")

        # 2. Oranlara göre sınır indekslerini hesapla
        train_end = int(total_count * RATIOS["train"])
        valid_end = train_end + int(total_count * RATIOS["valid"])

        # 3. Listeyi parçala
        splits = {
            "train": all_images[:train_end],
            "valid": all_images[train_end:valid_end],
            "test": all_images[valid_end:]
        }

        # 4. Kopyala ve İsimlendir (adult_001.jpg vb.)
        for split_name, img_list in splits.items():
            for i, img_path in enumerate(img_list):
                new_name = f"{cls}_{i:03d}{img_path.suffix}"
                shutil.copy(img_path, FINAL_DIR / split_name / cls / new_name)
        
        print(f"   🚀 {cls} bitti -> Train: {len(splits['train'])}, Valid: {len(splits['valid'])}, Test: {len(splits['test'])}")

if __name__ == "__main__":
    setup_folders()
    organize()
    print(f"\n✨ İşlem Tamam! Tüm resimlerin %70-20-10 kuralına göre '{FINAL_DIR}' adresinde organize edildi.")