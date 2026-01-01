import cv2

def draw_detections(frame, detections):
    """
    Detections: [[x1, y1, x2, y2, conf, cls_id], ...]
    """
    for det in detections:
        x1, y1, x2, y2, conf, _ = det
        
        # Renk (Yeşil)
        color = (0, 255, 0)
        
        # Bbox çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label hazırla
        label = f"Person {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Label arka planı (okunabilirlik için)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame

def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)