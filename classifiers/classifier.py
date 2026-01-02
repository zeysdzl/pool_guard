import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import sys

# Yolu ekle (ResNet dosyasını bulabilmesi için)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from classifiers.resnet_model import ResNetClassifier 

class PersonClassifier:
    def __init__(self, model_path, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🧠 Classifier (ResNet50) Başlatılıyor... | Cihaz: {self.device}")

        # 1. Modeli Kur (Eğitimdeki ayarların aynısı: freeze_backbone=False)
        self.model = ResNetClassifier(num_classes=2, freeze_backbone=False)
        
        # 2. Ağırlıkları Yükle
        if not os.path.exists(model_path):
            print(f"❌ HATA: Model dosyası bulunamadı: {model_path}")
            sys.exit(1)
            
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"✅ ResNet50 Ağırlıkları Yüklendi: {model_path}")
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            sys.exit(1)

        self.model.to(self.device)
        self.model.eval() 

        # 3. Transform (Eğitimdeki 'valid' ayarlarıyla BİREBİR AYNI olmalı)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # ResNet ImageNet standardı:
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.classes = ['adult', 'child']

    def predict(self, face_crop_bgr):
        """
        Girdi: OpenCV BGR formatında kırpılmış insan resmi
        Çıktı: (label, child_score)
        """
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return "unknown", 0.0

        # OpenCV BGR -> PIL RGB
        img_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Transform ve GPU'ya atma
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Child indexi 1'dir (folder yapısına göre 0=adult, 1=child)
            child_score = probabilities[0][1].item()
            
            # Karar Eşiği
            label = "child" if child_score > 0.5 else "adult"

        return label, child_score