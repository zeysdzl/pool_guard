from ultralytics import YOLO
import cv2
import numpy as np

class YoloDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25, optimize_for_pi: bool = False):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.optimize_for_pi = optimize_for_pi
        # Child/Adult model classes: 0=Adult, 1=Child
        self.target_classes = [0, 1]
        
        # Pi 5 optimization: Pre-allocate frame buffer
        if optimize_for_pi:
            self.target_size = (640, 480)  # Standard YOLO input size
            print("🍓 Raspberry Pi 5 optimization enabled")

    def detect(self, frame):
        """
        Frame alır, detection listesi döner.
        Return: list of [x1, y1, x2, y2, score, class_id, label]
        class_id: 0=Adult, 1=Child
        
        Optimized for Pi 5: Efficient resizing and memory management
        """
        # Pi 5 optimization: Resize early to reduce memory usage
        original_shape = frame.shape[:2]
        if self.optimize_for_pi and frame.shape[:2] != self.target_size[::-1]:
            # Use INTER_LINEAR for speed (faster than INTER_AREA for downscaling)
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Use half precision if available (Pi 5 doesn't have CUDA, but keeps code ready)
        results = self.model.predict(
            frame, 
            conf=self.conf_threshold, 
            verbose=False, 
            device='cpu' if self.optimize_for_pi else 0,  # Pi 5 uses CPU, otherwise GPU
            imgsz=640  # Standard YOLO input size
        )
        
        detections = []
        if not results:
            return detections

        # Get class names from model
        class_names = self.model.names
        
        # Scale factor if we resized (for Pi optimization)
        scale_x = original_shape[1] / frame.shape[1] if self.optimize_for_pi else 1.0
        scale_y = original_shape[0] / frame.shape[0] if self.optimize_for_pi else 1.0
        
        # Ultralytics results nesnesini parse et
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id in self.target_classes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Scale coordinates back to original frame size if resized
                if self.optimize_for_pi:
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                else:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                conf = float(box.conf[0])
                # Get label from model class names (normalize to lowercase)
                label = class_names.get(cls_id, "unknown").lower()
                detections.append([x1, y1, x2, y2, conf, cls_id, label])
        
        return detections