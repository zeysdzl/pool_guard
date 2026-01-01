import argparse
import cv2
import time
import os
import json
from detectors.yolo_detector import YoloDetector
from utils.drawing import draw_detections, draw_fps, draw_zone
from utils.geometry import is_point_in_polygon
from scripts.zone_editor import select_zone

# Kayıt dosyası yolu
ZONE_PATH = "zone.json"

def main():
    parser = argparse.ArgumentParser(description="Pool Guard MVP")
    parser.add_argument("--source", type=str, default="0", help="Webcam index")
    parser.add_argument("--weights", type=str, default="models/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    # --- KESİN ÇÖZÜM: Eski dosyayı fiziksel olarak sil ---
    if os.path.exists(ZONE_PATH):
        try:
            os.remove(ZONE_PATH)
            print("🧹 Eski bölge dosyası silindi (Temiz başlangıç).")
        except Exception as e:
            print(f"Uyarı: Dosya silinemedi: {e}")

    # 1. Kamerayı Aç
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Hata: Kamera açılamadı.")
        return

    # 2. Çizim Modunu Başlat (Dosya silindiği için mecbur açılacak)
    print("✏️  Çizim modu başlatılıyor...")
    zone_poly = select_zone(cap)
    
    # Eğer çizim yapmadan kapatırsa (boş dönerse)
    if not zone_poly or len(zone_poly) < 3:
        print("⚠️  Bölge çizilmedi! Program kapatılıyor.")
        cap.release()
        return

    print(f"✅ Yeni bölge kaydedildi ({len(zone_poly)} nokta). Tespit başlıyor...")

    # 3. Model Yükle
    print(f"Model: {args.weights}")
    detector = YoloDetector(args.weights, conf_threshold=args.conf)
    
    prev_time = 0
    
    # 4. Tespit Döngüsü
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Tespit
        raw_detections = detector.detect(frame)
        
        final_detections = []
        alarm_active = False
        
        for det in raw_detections:
            x1, y1, x2, y2, conf, cls_id = det
            
            # Ayak noktası (bbox alt orta)
            foot_point = ((x1 + x2) // 2, y2)
            
            # Bölge Kontrolü
            is_danger = False
            if zone_poly:
                is_danger = is_point_in_polygon(foot_point, zone_poly)
            
            if is_danger:
                alarm_active = True
            
            final_detections.append([x1, y1, x2, y2, conf, cls_id, is_danger])

        # Çizimler
        draw_zone(frame, zone_poly, alarm_active)
        draw_detections(frame, final_detections)
        
        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        draw_fps(frame, fps)

        cv2.imshow("Pool Guard", frame)
        
        # Gecikmeyi 50ms yaptık (Tuşları daha iyi algılasın diye)
        if cv2.waitKey(50) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()