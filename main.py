import argparse
import cv2
import time
import os
import numpy as np
import uuid # Benzersiz dosya ismi için

from detectors.yolo_detector import YoloDetector
from classifiers.classifier import PersonClassifier
from utils.drawing import draw_detections, draw_fps, draw_zone
from utils.geometry import is_point_in_polygon
from scripts.zone_editor import select_zone

ZONE_PATH = "zone.json"

def main():
    parser = argparse.ArgumentParser(description="Pool Guard - Active Learning Mode")
    parser.add_argument("--source", type=str, default="0", help="Webcam/Video")
    # Senin Eğittiğin Özel Model
    parser.add_argument("--weights", type=str, default="models/my_custom_yolov8.pt")
    # Keras'tan vazgeçip tekrar orijinal ResNet modeline dönüyoruz
    parser.add_argument("--cls_model", type=str, default="models/resnet50_best.pth")
    parser.add_argument("--conf", type=float, default=0.25)
    
    # 🔥 YENİ ÖZELLİK: Eğitim verisi toplama modu
    parser.add_argument("--collect_data", action="store_true", help="Tespit edilen kişileri kaydeder")
    args = parser.parse_args()

    # Klasör Hazırlığı
    if args.collect_data:
        os.makedirs("collected_data/unknown", exist_ok=True)
        print("📸 VERİ TOPLAMA MODU AKTİF: Resimler 'collected_data' klasörüne kaydedilecek.")

    # Kamera ve Video Kaynağı
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): return
    
    # Zone seçimi (Eski zone varsa temizle ve yenisini seçtir)
    if os.path.exists(ZONE_PATH):
        try: os.remove(ZONE_PATH)
        except: pass
    zone_poly = select_zone(cap)
    
    # Modellerin Yüklenmesi
    detector = YoloDetector(args.weights, conf_threshold=args.conf)
    classifier = PersonClassifier(model_path=args.cls_model)
    
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        raw_detections = detector.detect(frame)
        final_detections = []
        alarm_active = False
        
        for det in raw_detections:
            x1, y1, x2, y2, conf, cls_id = det
            
            # Koordinat ve Zone kontrolü
            foot_point = ((x1 + x2) // 2, y2)
            is_in_zone = is_point_in_polygon(foot_point, zone_poly) if zone_poly else False
            
            # Kırpma İşlemi (Görüntü sınırları dışına çıkmamak için max/min kontrolü)
            h, w, _ = frame.shape
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            
            label = "unknown"
            child_score = 0.0
            is_danger = False

            if cx2 > cx1 and cy2 > cy1:
                person_crop = frame[cy1:cy2, cx1:cx2]
                # ResNet Sınıflandırıcı tahmini
                label, child_score = classifier.predict(person_crop)
                
                # 🔥 VERİ KAYDETME (Active Learning)
                if args.collect_data and int(time.time() * 10) % 5 == 0:
                    unique_name = f"{label}_{uuid.uuid4().hex[:8]}.jpg"
                    save_path = os.path.join("collected_data", "unknown", unique_name)
                    cv2.imwrite(save_path, person_crop)

            # Alarm ve Tehlike Kontrolü
            if is_in_zone:
                if label == "child" and child_score > 0.5:
                    is_danger = True
                    alarm_active = True
            
            final_detections.append([x1, y1, x2, y2, label, child_score, is_danger])

        # Çizimler
        draw_zone(frame, zone_poly, alarm_active)
        draw_detections(frame, final_detections)
        
        # FPS Hesaplama
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        draw_fps(frame, fps)

        cv2.imshow("Pool Guard ResNet Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()