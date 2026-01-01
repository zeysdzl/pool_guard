import cv2
import numpy as np

def is_point_in_polygon(point, polygon):
    """
    point: (x, y)
    polygon: list of [x, y]
    Return: True if inside, False otherwise
    """
    # measureDist=False dönerse: +1 (içinde), -1 (dışında), 0 (kenarda)
    pts = np.array(polygon, np.int32)
    pts = pts.reshape((-1, 1, 2))
    result = cv2.pointPolygonTest(pts, point, False)
    return result >= 0