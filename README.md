# Pool Guard 🏊‍♂️ (MVP)

**Pool Guard**, havuz güvenliğini sağlamak amacıyla geliştirilen, bilgisayarlı görü (Computer Vision) tabanlı bir Python projesidir. Şu anki MVP (Minimum Viable Product) aşamasında, kamera görüntüsü üzerinden gerçek zamanlı insan tespiti (person detection) yapmaktadır.

## 🚀 Özellikler

* **Gerçek Zamanlı Tespit:** YOLOv8 (Ultralytics) kullanarak yüksek performanslı insan tespiti.
* **Donanım Desteği:** * NVIDIA GPU (CUDA) desteği (RTX serisi dahil).
    * CPU üzerinde optimize edilmiş çalışma modu.
* **Görselleştirme:** Canlı izleme penceresi, FPS sayacı ve bounding box çizimleri.
* **Kayıt:** Tespit anlarını video dosyası olarak kaydetme opsiyonu.

## 🛠️ Kurulum

1. **Repoyu Klonlayın:**
   ```bash
   git clone [https://github.com/zeysdzl/pool_guard.git](https://github.com/zeysdzl/pool_guard.git)
   cd pool_guard

2. Sanal Ortamı Oluşturun (Windows):
python -m venv .venv
.venv\Scripts\activate

3. Bağımlılıkları Yükleyin:
# Standart kurulum
pip install -r requirements.txt

# Eğer RTX 50 serisi (Blackwell) kullanıyorsanız (Özel PyTorch Sürümü):
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)

▶️ Kullanım
Uygulamayı başlatmak için ana dizinde:

PowerShell

# Webcam ile başlat (Varsayılan)
python main.py

# Video kaydı alarak başlat
python main.py --save

# Farklı bir model ile başlat (Örn: Eğitilmiş model)
python main.py --weights models/best.pt --conf 0.40

Çıkmak için q tuşuna basabilirsiniz.