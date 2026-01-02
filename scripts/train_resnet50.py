import os
import sys

# --- 1. KRİTİK ADIM: Önce Yolu Tanıt ---
# Bu kod bloğu, Python'a "projeni ana klasörden (pool_guard) gör" der.
# MUTLAKA diğer importlardan önce gelmelidir.
current_dir = os.path.dirname(os.path.abspath(__file__)) # scripts klasörü
parent_dir = os.path.dirname(current_dir) # pool_guard klasörü (ana dizin)
sys.path.append(parent_dir)

# --- 2. Şimdi Importları Yap ---
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Artık ana dizini gördüğü için bunu bulabilir:
try:
    from classifiers.resnet_model import ResNetClassifier
except ImportError as e:
    print("❌ HATA: 'classifiers' modülü bulunamadı!")
    print(f"Aranan yol: {parent_dir}")
    print("Lütfen 'classifiers/resnet_model.py' dosyasının adını ve yerini kontrol et.")
    sys.exit(1)

warnings.filterwarnings("ignore")

# --- AYARLAR ---
DATA_DIR = os.environ.get("DATASET_ROOT", "../pool_guard_data/classifier_final")
MODEL_OUT = os.environ.get("MODEL_OUT", "models")
SAVE_PATH = os.path.join(MODEL_OUT, "resnet50_best.pth")

os.makedirs(MODEL_OUT, exist_ok=True)

# ⚠️ BATCH SIZE AYARI (GPU'yu yormamak için düşük tuttuk)
BATCH_SIZE = 16 
EPOCHS = 15       
LEARNING_RATE = 0.0001 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    print(f"🚀 Transfer Learning Başlıyor (ResNet50) | Cihaz: {DEVICE}")
    print(f"⚠️  Batch Size: {BATCH_SIZE} (GPU koruma modu)")
    print(f"📂 Veri Yolu: {DATA_DIR}")

    # --- VERİ İŞLEME ---
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "valid": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    try:
        image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'valid']}
    except FileNotFoundError:
        print(f"❌ HATA: Veri klasörü bulunamadı: {DATA_DIR}")
        return

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes
    
    print(f"🏷️  Sınıflar: {class_names}")

    # --- MODELİ KUR ---
    model = ResNetClassifier(num_classes=len(class_names), freeze_backbone=False)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            pbar = tqdm(dataloaders[phase], desc=f"   {phase.upper()}")
            
            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f"   Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"   ✅ Model Kaydedildi: {best_acc:.4f}")

    print(f"\n🎉 ResNet Eğitimi Bitti! En iyi skor: %{best_acc*100:.2f}")

if __name__ == "__main__":
    train_model()