import cv2
import numpy as np
from typing import Tuple, Dict, List

from Detection.Geometry import GeometryUtils, AffineTransformer
from Detection.ArUco import MarkersDetector, MarkerData, Direction
from Detection import BoardCalibrator

class CalibrationPipeline:
    _grid: Dict[Tuple[float,float],Tuple[float,float]] = None
    _transform_params: np.ndarray = None
    _width = 0
    _height = 0

    def __init__(self):
        self.detector = MarkersDetector()
        self.calibrator = BoardCalibrator(self.detector)
        self.transformer = AffineTransformer()
        self._grid: Dict[Tuple[float,float],Tuple[float,float]] = None
        self._transform_params: np.ndarray = None
        self._width = 0
        self._height = 0

    def process_image(self, image: np.ndarray) -> None:
        detection = self.detector.detect(image)

        if detection.ids is None:
            print("Маркеры не обнаружены")
            return

        # Поиск общего ограничивающего прямоугольника
        all_corners=[]
        for corner in detection.corners:
            for point in corner[0]:
                all_corners.append(point)
        all_corners = np.array(all_corners)

        board_rect = GeometryUtils.get_min_area_rect_points(all_corners)

        # Выбор трех ключевых углов
        source_corners = np.array([board_rect[1], board_rect[2], board_rect[0]])

        # Перспективная коррекция
        self._transform_params, self._width, self._height = self.transformer.get_transform_params(source_corners)
        warped = self.transformer.transform_board(image, self._transform_params, self._width, self._height)

        # Расчет сетки
        cols, rows = self.calibrator.calculate_grid_dimensions(warped)
        self._grid = self.calibrator.calculate_coordinate_grid(rows, cols)

        print(f"Calibration Finished, grid size: {len(cols)}x{len(rows)}")
        return self._grid != {}

    def get_board_data(self, image: np.ndarray) -> List[MarkerData]:
        warped = self.transformer.transform_board(image, self._transform_params, self._width, self._height)
        detection = self.detector.detect(warped)
        result = []
        
        for index,cord in enumerate(detection.corners):
            tmp = warped.copy()
            grid = self.find_closest_grid(cord)

            center = cord[0].mean(axis=0)
            cv2.circle(tmp, center.astype(int), 10, (255,0,0), 3)
            alignment = GeometryUtils.get_rotation_degree(cord[0][0], center)

            data = MarkerData(detection.ids[index], grid, alignment)
            result.append(data)
            # print(data)
            # cv2.aruco.drawDetectedMarkers(tmp, detection.corners, detection.ids)

            # # visualized
            # cv2.imshow('Calibrated', tmp)
            # cv2.waitKey()
        return result
    
    def find_closest_grid(self, coordinate: np.ndarray) -> Tuple[int, int]:
        center = coordinate[0].mean(axis=0)
        min_dist = 1e10
        closest_grid=(1e10,1e10)
        # print(center)

        for cord in self._grid:
            dist = GeometryUtils.calculate_distance(cord,center)
            if dist < min_dist:
                min_dist = dist
                closest_grid = self._grid[cord]
        
        return closest_grid