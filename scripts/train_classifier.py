import os
import warnings
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Environment & warnings
# -----------------------------------------------------------------------------
load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Configuration (NO HARDCODED PATHS)
# -----------------------------------------------------------------------------
DATA_DIR = os.environ["DATASET_ROOT"]          # e.g. ../pool_guard_data/classifier
MODEL_OUT = os.environ["MODEL_OUT"]             # e.g. ../pool_guard_models
SAVE_PATH = os.path.join(MODEL_OUT, "best_classifier.pth")
os.makedirs(MODEL_OUT, exist_ok=True)

BATCH_SIZE = 64        # RTX 5070 -> OK
EPOCHS = 15
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_model() -> None:
    print(f"Eğitim cihazı: {DEVICE}")

    # -------------------------------------------------------------------------
    # Data transforms
    # -------------------------------------------------------------------------
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]),
        "valid": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]),
    }

    # -------------------------------------------------------------------------
    # Datasets & loaders (Windows-safe)
    # -------------------------------------------------------------------------
    image_datasets = {
        split: datasets.ImageFolder(
            os.path.join(DATA_DIR, split),
            data_transforms[split],
        )
        for split in ["train", "valid"]
    }

    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        for split in ["train", "valid"]
    }

    class_names = image_datasets["train"].classes
    print(f"Sınıflar: {class_names}")

    # -------------------------------------------------------------------------
    # Model definition (MobileNetV3 Small)
    # -------------------------------------------------------------------------
    model = models.mobilenet_v3_small(weights="DEFAULT")

    # 🔒 Freeze backbone (CRITICAL for small datasets)
    for param in model.features.parameters():
        param.requires_grad = False

    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, 2)
    model = model.to(DEVICE)

    # -------------------------------------------------------------------------
    # Loss & optimizer (classifier head only)
    # -------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
    )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    print("\n--- Eğitim Başlıyor ---")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            pbar = tqdm(dataloaders[phase], desc=f"[{phase}]")
            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

                pbar.set_postfix(loss=f"{loss.item():.4f}")

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f"{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"✅ Yeni en iyi model kaydedildi: {best_acc:.4f}")

    print(f"\nEğitim tamamlandı. En iyi doğruluk: {best_acc:.4f}")
    print(f"Model yolu: {SAVE_PATH}")


# -----------------------------------------------------------------------------
# Windows entry-point guard
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train_model()
