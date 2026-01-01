import cv2

def draw_detections(frame, detections):
    """
    detections: [[x1, y1, x2, y2, conf, cls_id, is_danger], ...]
    """
    for det in detections:
        # Listeden değerleri al (artık is_danger parametresi de geliyor)
        x1, y1, x2, y2, conf, cls_id, is_danger = det
        
        # Tehlike varsa KIRMIZI, yoksa YEŞİL
        color = (0, 0, 255) if is_danger else (0, 255, 0)
        
        # Kutu çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Etiket hazırla
        label = f"Person {conf:.2f}"
        if is_danger:
            label += " [ALARM]"
            
        # Etiket arka planı ve yazı
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_zone(frame, zone_poly, is_alarm_active):
    """
    Bölge sınırlarını çizer.
    """
    if not zone_poly:
        return

    import numpy as np
    pts = np.array(zone_poly, np.int32)
    
    # Alarm varsa çizgi kalınlaşır ve parlar
    color = (0, 0, 255) if is_alarm_active else (255, 0, 0) # Kırmızı veya Mavi
    thickness = 4 if is_alarm_active else 2
    
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    
    if is_alarm_active:
        cv2.putText(frame, "!!! DANGER ZONE ALERT !!!", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)