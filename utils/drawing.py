import cv2
import numpy as np

def draw_detections(frame, detections):
    """
    Beklenen format: [[x1, y1, x2, y2, label, child_score, is_danger], ...]
    """
    for det in detections:
        x1, y1, x2, y2, label, score, is_danger = det
        
        # Renk Ayarı
        if is_danger:
            color = (0, 0, 255)       # Kırmızı (TEHLİKE)
        elif label == "child":
            color = (0, 255, 255)     # Sarı (Çocuk - Güvenli Bölgede)
        else:
            color = (0, 255, 0)       # Yeşil (Yetişkin)
        
        # Kutuyu Çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Etiketi Hazırla
        text = f"{label.upper()} %{score*100:.0f}"
        if is_danger:
            text = "!!! ALARM !!!"
            
        # Arka plan ve Yazı
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
        
        # Yazı rengi (Sarı üzerine siyah, diğerlerine beyaz)
        text_color = (0, 0, 0) if (label == "child" and not is_danger) else (255, 255, 255)
        
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

# ... (draw_zone ve draw_fps aynı kalabilir) ...
def draw_zone(frame, zone_poly, is_alarm_active):
    if not zone_poly: return
    pts = np.array(zone_poly, np.int32)
    color = (0, 0, 255) if is_alarm_active else (255, 0, 0)
    thickness = 4 if is_alarm_active else 2
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    if is_alarm_active:
        cv2.putText(frame, "TEHLIKELI GIRIS!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)