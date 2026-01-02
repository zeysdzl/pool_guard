import argparse
import cv2
import time
import os
import numpy as np

from detectors.yolo_detector import YoloDetector
from classifiers.classifier import PersonClassifier
from utils.drawing import draw_detections, draw_fps, draw_zone
from utils.geometry import is_point_in_polygon
from scripts.zone_editor import select_zone

ZONE_PATH = "zone.json"

def main():
    parser = argparse.ArgumentParser(description="Pool Guard - Final System")
    parser.add_argument("--source", type=str, default="0", help="Webcam index")
    
    # --- KRİTİK DÜZELTME: Senin özel YOLO modelin ---
    parser.add_argument("--weights", type=str, default="models/person_yolov8n.pt", help="Detection Model")
    
    # --- Yeni Eğittiğimiz Güçlü ResNet ---
    parser.add_argument("--cls_model", type=str, default="models/resnet50_best.pth", help="Classification Model")
    
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    # --- Temizlik ---
    if os.path.exists(ZONE_PATH):
        try: os.remove(ZONE_PATH)
        except: pass

    # 1. Kamera ve Bölge
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Hata: Kamera açılamadı.")
        return
    
    print("✏️  Güvenlik Bölgesini Çiziniz...")
    zone_poly = select_zone(cap)
    if not zone_poly or len(zone_poly) < 3:
        return

    # 2. Modelleri Yükle
    print(f"🔹 Dedektör Yükleniyor: {args.weights}")
    if not os.path.exists(args.weights):
        print(f"❌ HATA: {args.weights} bulunamadı! Lütfen dosya adını kontrol et.")
        return
    detector = YoloDetector(args.weights, conf_threshold=args.conf)
    
    print(f"🔹 Sınıflandırıcı Yükleniyor: {args.cls_model}")
    if not os.path.exists(args.cls_model):
        print(f"❌ HATA: {args.cls_model} bulunamadı!")
        return
    classifier = PersonClassifier(model_path=args.cls_model)

    print("🚀 SİSTEM BAŞLATILDI! (Person YOLO + ResNet50)")
    prev_time = 0

    # 3. Ana Döngü
    while True:
        ret, frame = cap.read()
        if not ret: break

        # A. İnsanları Bul (Detect)
        raw_detections = detector.detect(frame)
        
        final_detections = []
        alarm_active = False
        
        for det in raw_detections:
            x1, y1, x2, y2, conf, cls_id = det
            
            foot_point = ((x1 + x2) // 2, y2)
            
            # B. Bölgede mi?
            is_in_zone = False
            if zone_poly:
                is_in_zone = is_point_in_polygon(foot_point, zone_poly)
            
            # C. Sınıflandır (Classify)
            label = "unknown"
            child_score = 0.0
            is_danger = False
            
            # Resmi Kırp ve ResNet'e Sor
            h, w, _ = frame.shape
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            
            if cx2 > cx1 and cy2 > cy1:
                person_crop = frame[cy1:cy2, cx1:cx2]
                label, child_score = classifier.predict(person_crop)
            
            # D. ALARM MANTIĞI 🚨
            # Eğer Bölgedeyse VE (Çocuksa VEYA Model %50'den fazla eminse)
            if is_in_zone:
                if label == "child" and child_score > 0.5:
                    is_danger = True
                    alarm_active = True
            
            final_detections.append([x1, y1, x2, y2, label, child_score, is_danger])

        # Çizimler
        draw_zone(frame, zone_poly, alarm_active)
        draw_detections(frame, final_detections)
        
        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        draw_fps(frame, fps)

        cv2.imshow("Pool Guard Final", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()