import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # --- BLOK 1: Görüntüden Özellik Çıkarma ---
        # Keras: Conv2D(32, (3,3), activation='relu')
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2) # Boyutu yarıya indirir
        
        # --- BLOK 2 ---
        # Keras: Conv2D(64, (3,3), activation='relu')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # --- BLOK 3 ---
        # Keras: Conv2D(128, (3,3), activation='relu')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # --- SINIFLANDIRMA (Dense Layers) ---
        # Giriş boyutu hesabına göre (224x224 girerse 3 kez pool sonrası 28x28 kalır)
        # 128 kanal * 28 * 28 piksel
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [Batch, 3, 224, 224]
        
        # 1. Blok
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 2. Blok
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 3. Blok
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Düzleştir (Flatten)
        x = x.view(-1, 128 * 28 * 28)
        
        # Sınıflandır
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Softmax burada yok, Loss fonksiyonunda var
        return x