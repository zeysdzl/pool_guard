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

def distance_to_polygon_boundary(point, polygon):
    """
    Calculate the minimum distance from a point to the polygon boundary.
    point: (x, y)
    polygon: list of [x, y]
    Return: distance in pixels (positive if outside, negative if inside)
    """
    if not polygon or len(polygon) < 3:
        return float('inf')
    
    pts = np.array(polygon, np.int32)
    pts = pts.reshape((-1, 1, 2))
    # measureDist=True returns signed distance: +ve outside, -ve inside
    distance = cv2.pointPolygonTest(pts, point, True)
    return abs(distance)  # Return absolute distance to boundary