import torch
import torch.nn as nn
from torchvision import models

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super(ResNetClassifier, self).__init__()
        
        # 1. Dev Beyni İndir (ResNet50)
        # 'weights="DEFAULT"' diyerek ImageNet ile eğitilmiş halini alıyoruz.
        self.model = models.resnet50(weights='DEFAULT')

        # 2. (Opsiyonel) Bilgileri Dondur
        # Modelin ilk katmanları (kenar, köşe tanıma) zaten mükemmel.
        # Onları dondurup sadece son katmanları eğitirsek çok hızlı sonuç alırız.
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # 3. Son Katmanı Değiştir (Ameliyat)
        # ResNet'in orijinali 1000 sınıf tanır. Biz bunu söküp 2 sınıf (Adult/Child) takacağız.
        num_ftrs = self.model.fc.in_features
        
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3), # Ezberlemeyi önle
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)