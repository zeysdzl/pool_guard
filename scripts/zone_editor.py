import cv2
import json
import numpy as np
import os

ZONE_FILE = "zone.json"
points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Sol Tık: Nokta Ekle
        points.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Sağ Tık: Bitir
        param[0] = True 

def select_zone(cap):
    global points
    points = [] # Listeyi sıfırla
    
    cv2.namedWindow("Zone Editor")
    finished = [False] 
    cv2.setMouseCallback("Zone Editor", mouse_callback, finished)

    print("--- ÇİZİM MODU ---")
    print("[SOL TIK] : Nokta Ekle")
    print("[SAĞ TIK] : Kaydet ve Başla")
    print("[R]       : Temizle")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Çizim
        if len(points) > 1:
            pts = np.array(points, np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            
        for p in points:
            cv2.circle(frame, tuple(p), 5, (0, 255, 255), -1)

        cv2.putText(frame, "Bitirmek icin SAG TIKLA", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Zone Editor", frame)
        
        # BURASI ÖNEMLİ: Tuş algılama süresini 50ms yaptık
        key = cv2.waitKey(50) & 0xFF

        # Sağ tık VEYA 's' tuşu
        if finished[0] or key == ord('s'):
            if len(points) < 3:
                print("⚠️ En az 3 nokta gerekli!")
                finished[0] = False
                continue
            
            with open(ZONE_FILE, 'w') as f:
                json.dump(points, f)
            break
        
        elif key == ord('r'):
            points = []
            print("Çizim temizlendi.")
        
        elif key == ord('q'):
            print("İptal edildi.")
            points = []
            break

    cv2.destroyWindow("Zone Editor") # Sadece editör penceresini kapat
    return points

if __name__ == "__main__":
    # Test için
    cap = cv2.VideoCapture(0)
    select_zone(cap)