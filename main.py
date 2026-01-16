import argparse
import cv2
import time
import os
import numpy as np
import uuid # Benzersiz dosya ismi için

from detectors.yolo_detector import YoloDetector
from utils.drawing import draw_detections, draw_fps, draw_zone
from utils.geometry import is_point_in_polygon, distance_to_polygon_boundary
from scripts.zone_editor import select_zone

ZONE_PATH = "zone.json"

def main():
    parser = argparse.ArgumentParser(description="Pool Guard - YOLO Child/Adult Detection")
    parser.add_argument("--source", type=str, default="0", help="Webcam/Video")
    # YOLO Child/Adult Detection Model
    parser.add_argument("--weights", type=str, default="models/YOLOV5S_child_adult.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    
    # 🔥 YENİ ÖZELLİK: Eğitim verisi toplama modu
    parser.add_argument("--collect_data", action="store_true", help="Tespit edilen kişileri kaydeder")
    args = parser.parse_args()

    # Klasör Hazırlığı
    if args.collect_data:
        os.makedirs("collected_data/to_label", exist_ok=True)
        print("📸 ACTIVE LEARNING MODU AKTİF:")
        print("   - Belirsiz tespitler (0.25-0.60 güven) kaydedilecek")
        print("   - Zone sınırına yakın çocuklar kaydedilecek")
        print("   - Resimler 'collected_data/to_label/' klasörüne kaydedilecek")

    # Kamera ve Video Kaynağı
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): return
    
    # Zone seçimi (Eski zone varsa temizle ve yenisini seçtir)
    if os.path.exists(ZONE_PATH):
        try: os.remove(ZONE_PATH)
        except: pass
    zone_poly = select_zone(cap)
    
    # YOLO Model Yüklenmesi (Child/Adult Detection)
    # Enable Pi 5 optimization if running on Raspberry Pi
    optimize_pi = os.environ.get('RASPBERRY_PI', '').lower() == 'true'
    detector = YoloDetector(args.weights, conf_threshold=args.conf, optimize_for_pi=optimize_pi)
    print("✅ YOLO Child/Adult Detection Model Yüklendi")
    
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        raw_detections = detector.detect(frame)
        final_detections = []
        alarm_active = False
        
        for det in raw_detections:
            x1, y1, x2, y2, conf, cls_id, label = det
            
            # Koordinat ve Zone kontrolü
            foot_point = ((x1 + x2) // 2, y2)
            is_in_zone = is_point_in_polygon(foot_point, zone_poly) if zone_poly else False
            
            is_danger = False

            # 🔥 ENHANCED ACTIVE LEARNING - Smart Data Collection
            should_save = False
            save_reason = ""
            
            if args.collect_data and zone_poly:
                h, w, _ = frame.shape
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(w, x2), min(h, y2)
                
                if cx2 > cx1 and cy2 > cy1:
                    # Check 1: Uncertain detections (confidence between 0.25 and 0.60)
                    if 0.25 <= conf <= 0.60:
                        should_save = True
                        save_reason = "uncertain"
                    
                    # Check 2: Child detected near zone boundary (within 50 pixels)
                    elif label == "child":
                        boundary_dist = distance_to_polygon_boundary(foot_point, zone_poly)
                        if boundary_dist <= 50:  # Within 50 pixels of boundary
                            should_save = True
                            save_reason = "boundary"
                    
                    # Save with metadata in filename
                    if should_save:
                        # Throttle: Save max 1 image per second per detection
                        current_time = int(time.time())
                        unique_name = f"{label}_{conf:.2f}_{save_reason}_{current_time}_{uuid.uuid4().hex[:6]}.jpg"
                        save_path = os.path.join("collected_data", "to_label", unique_name)
                        
                        # Efficient crop (no copy, direct slice)
                        person_crop = frame[cy1:cy2, cx1:cx2]
                        cv2.imwrite(save_path, person_crop)

            # Alarm ve Tehlike Kontrolü - Use YOLO confidence directly
            if is_in_zone:
                if label == "child" and conf > 0.5:
                    is_danger = True
                    alarm_active = True
            
            # Use YOLO confidence score directly (not child_score)
            final_detections.append([x1, y1, x2, y2, label, conf, is_danger])

        # Çizimler
        draw_zone(frame, zone_poly, alarm_active)
        draw_detections(frame, final_detections)
        
        # FPS Hesaplama
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        draw_fps(frame, fps)

        cv2.imshow("Pool Guard - YOLO Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()