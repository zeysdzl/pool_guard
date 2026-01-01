from ultralytics import YOLO
import cv2

class YoloDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        # Model yoksa otomatik yolov8n.pt indirir (demo için hayat kurtarır)
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # Sadece insan sınıfı (COCO dataset'te class_id=0)
        self.target_classes = [0] 

    def detect(self, frame):
        """
        Frame alır, detection listesi döner.
        Return: list of [x1, y1, x2, y2, score, class_id]
        """
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False, device=0)
        
        detections = []
        if not results:
            return detections

        # Ultralytics results nesnesini parse et
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id in self.target_classes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append([int(x1), int(y1), int(x2), int(y2), conf, cls_id])
        
        return detections