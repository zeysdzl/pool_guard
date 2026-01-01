import argparse
import cv2
import time
import os
from detectors.yolo_detector import YoloDetector
from utils.drawing import draw_detections, draw_fps

def main():
    # 1. Argümanları al
    parser = argparse.ArgumentParser(description="Pool Guard MVP - Person Detection")
    parser.add_argument("--source", type=str, default="0", help="Webcam index or video path")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", action="store_true", help="Save output video to runs/")
    args = parser.parse_args()

    # 2. Kaynağı hazırla
    source = args.source
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Hata: Video kaynağı açılamadı -> {source}")
        return

    # 3. Modeli yükle
    print(f"Model yükleniyor: {args.weights}...")
    detector = YoloDetector(args.weights, conf_threshold=args.conf)

    # 4. Video Yazıcı (Opsiyonel)
    out = None
    if args.save:
        os.makedirs("runs", exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
        out = cv2.VideoWriter("runs/output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (width, height))
        print("Kayıt başladı: runs/output.mp4")

    # 5. Ana Döngü
    print("Başlatıldı. Çıkmak için 'q' tuşuna basın.")
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detection
        detections = detector.detect(frame)

        # Drawing
        draw_detections(frame, detections)
        
        # FPS Hesapla & Çiz
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        draw_fps(frame, fps)

        # Göster
        cv2.imshow("Pool Guard MVP", frame)
        
        # Kaydet
        if out:
            out.write(frame)

        # Çıkış kontrolü
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Temizlik
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("Kapanıyor...")

if __name__ == "__main__":
    main()