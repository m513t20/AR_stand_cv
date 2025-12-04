import pygame
import sys

from SnakeObjects import SnakeGrid, CellType, Snake
from Base import BaseGrid

def start_snake(data: str,rows: int = 8, cols: int = 8):
    pygame.init()

    pygame.display.set_caption("Змейка")
    clock = pygame.time.Clock()

    grid = SnakeGrid(rows, cols, data)
    snake = Snake(rows, cols)
    screen = pygame.display.set_mode((grid.matrix.shape[1]*grid.cell_size, grid.matrix.shape[0]*grid.cell_size))

    while True:
        process_events(snake)
        snake.update()

        x, y = snake.get_head_position()
        process_field(y,x,grid,snake)
        # TODO
        # reaction.dispatch_reaction(snake, grid.snake_matrix[y, x].cell_type)
        if grid.food_amount == 0:
            print("YAY")
            break
        
        grid.draw_grid(screen)
        snake.render(screen, grid.cell_size)
        pygame.display.update()    
        clock.tick(1)  # Скорость игры


def process_events(snake: Snake):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            # Управление стрелками
            if event.key == pygame.K_UP and snake.direction != (0, 1):
                snake.direction = (0, -1)
            elif event.key == pygame.K_DOWN and snake.direction != (0, -1):
                snake.direction = (0, 1)
            elif event.key == pygame.K_LEFT and snake.direction != (1, 0):
                snake.direction = (-1, 0)
            elif event.key == pygame.K_RIGHT and snake.direction != (-1, 0):
                snake.direction = (1, 0)


def process_field(y: int, x:int, grid: SnakeGrid, snake: Snake):
    if grid.snake_matrix[y, x].cell_type == CellType.WALL:
        snake.reset()
    elif grid.snake_matrix[y, x].cell_type == CellType.UTURN:
        snake.positions.reverse()
        snake.direction = (-snake.direction[0], -snake.direction[1])
    elif grid.snake_matrix[y, x].cell_type == CellType.TURN:
        match grid.matrix[y,x].turn:
            case 0:
                snake.direction = (0, -1)
                
            case 1:
                snake.direction = (1, 0)
                
            case 2: 
                snake.direction = (0, 1)
                
            case 3:
                snake.direction = (-1, 0)
                                    
    elif grid.snake_matrix[y, x].cell_type == CellType.FOOD:
        snake.length += 1
        snake.score += 1
        grid.food_amount -= 1
        grid.snake_matrix[y, x].cell_type = CellType.NOTHING