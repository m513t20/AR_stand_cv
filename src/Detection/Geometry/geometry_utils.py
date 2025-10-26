import cv2
import numpy as np

from Detection.ArUco import Direction

class GeometryUtils:
    @staticmethod
    def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        return np.linalg.norm(point1 - point2)

    @staticmethod
    def get_min_area_rect_points(points: np.ndarray) -> np.ndarray:
        rect = cv2.minAreaRect(points)
        return cv2.boxPoints(rect)
    
    @staticmethod
    def get_rotation_degree(top_left: np.ndarray, center: np.ndarray) -> Direction:
        center_aligned_cords = top_left - center
        if center_aligned_cords[0] > 0 and center_aligned_cords[1] > 0:
            return Direction.TURNED_180_DEG
        elif center_aligned_cords[0] < 0 and center_aligned_cords[1] > 0:
            return Direction.TURNED_270_DEG
        elif center_aligned_cords[0] < 0 and center_aligned_cords[1] < 0:
            return Direction.NOT_TURNED
        else:
            return Direction.TURNED_90_DEG
