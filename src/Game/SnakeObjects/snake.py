import pygame

SNAKE_HEAD = (0, 255, 0)
SNAKE_BODY = (0, 200, 0)

class Snake:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols 
        self.positions = [(self.cols // 2, self.rows // 2)]  # Начальная позиция
        self.direction = (1, 0)  # Начальное направление
        self.length = 1
        self.score = 0

    def get_head_position(self) -> tuple[int, int]:
        return self.positions[0]

    def update(self):
        head = self.get_head_position()
        x, y = self.direction
        new_x = (head[0] + x) % self.cols  
        new_y = (head[1] + y) % self.rows
        new_position = (new_x, new_y)

        if new_position in self.positions[1:]:
            self.reset()  # Рестарт при столкновении с собой
        else:
            self.positions.insert(0, new_position)
            if len(self.positions) > self.length:
                self.positions.pop()

    def reset(self):
        self.positions = [(self.cols // 2, self.rows // 2)]
        self.direction = (1, 0)
        self.length = 1
        self.score = 0

    def render(self, screen: pygame.Surface, cell_size: int):
        # Рисуем тело змейки
        for i, p in enumerate(self.positions):
            color = SNAKE_HEAD if i == 0 else SNAKE_BODY
            rect = pygame.Rect(p[0] * cell_size, p[1] * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect)