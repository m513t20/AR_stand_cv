import random
import numpy as np
import pygame

from dataclasses import dataclass
from enum import Enum, auto

from Base import BaseGrid, GridCell

BACKGROUND = (15, 15, 15)

class CellType(Enum):
    NOTHING = -1
    FOOD = auto()
    WALL = auto()
    UTURN = auto()
    TURN = auto()


# Цвета
COLORS = {
    CellType.NOTHING : (40, 40, 40),
    CellType.FOOD : (255, 50, 50),
    CellType.TURN : (50, 50, 250),
    CellType.UTURN : (250, 50, 250),
    CellType.WALL : (255, 255, 250),
}

@dataclass
class SnakeCell:
    cell_type: CellType
    base_data: GridCell

class SnakeGrid(BaseGrid): 

    def __init__(self, rows: int, cols: int, matrix: str, cell_size = 40):
        super().__init__(rows, cols, matrix)
        self.snake_matrix = self.matrix.copy()
        
        self.blocked_matrix = np.zeros_like(self.snake_matrix)
        self.blocked_matrix[
            self.blocked_matrix.shape[0] // 2 - 2 : self.blocked_matrix.shape[0] // 2 + 2,
            self.blocked_matrix.shape[1] // 2 - 2 : self.blocked_matrix.shape[1] // 2 + 2,    
        ] = 1

        self.create_food()
        self.create_wall()
        self.cell_size = cell_size
        for x in range(0, self.matrix.shape[1]):
            for y in range(0, self.matrix.shape[0]):
                if isinstance(self.snake_matrix[y, x], SnakeCell):
                    continue
                elif self.blocked_matrix[y, x] == 1:
                    self.snake_matrix[y, x] = SnakeCell(CellType.NOTHING, self.matrix[y, x])
                    continue
                cell_type = CellType(self.matrix[y, x].id)
                self.snake_matrix[y, x] = SnakeCell(cell_type, self.matrix[y, x])

    def _randomize_position_set(self, amount: int):
        res = []
        while len(res) < amount:
            position = (random.randint(0, self.matrix.shape[0] - 1),
                            random.randint(0, self.matrix.shape[1] - 1))
            if not position in res and self.blocked_matrix[position[0], position[1]] == 0:
                res.append(position)
        return res
    
    def create_food(self, amount: int = 4):
        # расположение еды
        foods = self._randomize_position_set(amount)
        self.food_amount = len(foods)
        for food in foods:
            self.blocked_matrix[food[0], food[1]] = 1
            self.snake_matrix[food[0], food[1]] = SnakeCell(CellType.FOOD, self.matrix[food[1], food[0]])

    def create_wall(self, amount: int = 10):
        # расположение стен
        walls = self._randomize_position_set(amount)
        self.wall_amount = len(walls)
        for wall in walls:
            self.blocked_matrix[wall[0], wall[1]] = 1
            self.snake_matrix[wall[0], wall[1]] = SnakeCell(CellType.WALL, self.matrix[wall[1], wall[0]])

    def update_grid(self, matrix: BaseGrid):
        for x in range(0, matrix.matrix.shape[1]):
            for y in range(0, matrix.matrix.shape[0]):
                if self.blocked_matrix[y, x] == 1:
                    continue
                cell_type = CellType(matrix.matrix[y, x].id)
                self.snake_matrix[y, x] = SnakeCell(cell_type, matrix.matrix[y, x])

    def draw_grid(self, screen: pygame.Surface):
        screen.fill(BACKGROUND)
        for x in range(0, self.matrix.shape[1]):
            for y in range(0, self.matrix.shape[0]):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(screen, (40, 40, 40), rect, 1)
                if self.snake_matrix[y, x].cell_type != CellType.NOTHING:
                    pygame.draw.rect(screen, COLORS[self.snake_matrix[y, x].cell_type], rect)
