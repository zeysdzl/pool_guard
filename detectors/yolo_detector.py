from ultralytics import YOLO
import cv2

class YoloDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # Child/Adult model classes: 0=Adult, 1=Child
        self.target_classes = [0, 1] 

    def detect(self, frame):
        """
        Frame alır, detection listesi döner.
        Return: list of [x1, y1, x2, y2, score, class_id, label]
        class_id: 0=Adult, 1=Child
        """
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False, device=0)
        
        detections = []
        if not results:
            return detections

        # Get class names from model
        class_names = self.model.names
        
        # Ultralytics results nesnesini parse et
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id in self.target_classes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                # Get label from model class names (normalize to lowercase)
                label = class_names.get(cls_id, "unknown").lower()
                detections.append([int(x1), int(y1), int(x2), int(y2), conf, cls_id, label])
        
        return detections