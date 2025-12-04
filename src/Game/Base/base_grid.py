import json
import numpy as np

from dataclasses import dataclass

@dataclass
class GridCell:
    id: int
    turn: int
    
class BaseGrid:
    
    def __init__(self, rows: int, cols: int, matrix: str):
        data = json.loads(matrix)
        self.matrix = np.array([[GridCell(-1, 0) for _ in range(cols)] for _ in range(rows)])    

        for cur_data in data["matrix"]:
            cords =  cur_data["cords"]
            id = cur_data["id"]
            turn = cur_data["turn"]
            
            self.matrix[cords[1], cords[0]] = GridCell(id, turn)