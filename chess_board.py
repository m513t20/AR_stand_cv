import pygame
import requests
import time
import sys

# --- КОНСТАНТЫ ---
TILE_SIZE = 70
GAP = 4
COLS = 10  # 0 и 9 - боковые панели, 1-8 - доска
ROWS = 8
WIDTH = COLS * TILE_SIZE
HEIGHT = ROWS * TILE_SIZE
FPS = 30

# Цвета
COLOR_BG = (40, 40, 40)              
COLOR_WHITE = (240, 217, 181)
COLOR_BLACK = (181, 136, 99)
COLOR_SIDEBAR_WHITE = (200, 200, 200)
COLOR_SIDEBAR_BLACK = (50, 50, 50)
COLOR_HIGHLIGHT = (100, 200, 100, 150) # Зеленый для ходов
COLOR_CHECK = (200, 50, 50, 150)       # Красный для шаха
COLOR_DESYNC = (255, 150, 0, 150)      # Оранжевый для рассинхрона

# Маппинг ID фигур в Unicode символы для отрисовки без картинок
PIECES_UNICODE = {
    "white": {1: "♙", 2: "♘", 3: "♗", 4: "♖", 5: "♕", 6: "♔"},
    "black": {1: "♟", 2: "♞", 3: "♝", 4: "♜", 5: "♛", 6: "♚"}
}

BASE_URL = "http://127.0.0.1:8001"

# --- СЕТЕВОЙ МОДУЛЬ ---
class NetworkManager:
    def get_start_state(self, reset):
        try:
            if reset:
                requests.post(f"{BASE_URL}/reset", timeout=5)
            response = requests.get(f"{BASE_URL}/data", timeout=5)
            return response.json()            
        except requests.exceptions.RequestException as e:
            print(f"Ошибка сети (GET): {e}")
            return None

    def get_board_state(self):
        try:
            response = requests.get(f"{BASE_URL}/step", timeout=5)
            return response.json()            
        except requests.exceptions.RequestException as e:
            print(f"Ошибка сети (GET): {e}")
            return None

class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont("Arial", int(TILE_SIZE * 0.7))
        self.small_font = pygame.font.SysFont("Arial", 20)

    def draw_board(self, state):
        self.screen.fill((0, 0, 0))

        # 1. Отрисовка боковых панелей
        is_white_turn = state.get("is_white_turn", True) if state else True
        turn_color = COLOR_SIDEBAR_WHITE if is_white_turn else COLOR_SIDEBAR_BLACK
        
        pygame.draw.rect(self.screen, turn_color, (0, 0, TILE_SIZE, HEIGHT))
        pygame.draw.rect(self.screen, turn_color, (9 * TILE_SIZE, 0, TILE_SIZE, HEIGHT))

        # 2. Отрисовка шахматной доски
        for logical_row in range(ROWS):
            visual_row = 7 - logical_row
            
            for col_10x8 in range(1, 9):
                visual_col = logical_row + 1
                visual_row = col_10x8 - 1

                is_light_square = (logical_row + (col_10x8 - 1)) % 2 == 0
                color = COLOR_WHITE if not is_light_square else COLOR_BLACK

                rect_x = visual_col * TILE_SIZE + GAP // 2
                rect_y = visual_row * TILE_SIZE + GAP // 2
                rect_size = TILE_SIZE - GAP
                
                pygame.draw.rect(self.screen, color, (rect_x, rect_y, rect_size, rect_size))

        if not state:
            return

        # 3. Подсветка статусов (Шах, Ходы, Рассинхрон)
        statuses = state.get("status", [])
        for status in statuses:
            desc = status.get("description", "")
            if desc not in ["availble moves", "check", "desync"]:
                continue

            sq = status.get("square") or status.get("Square")
            if not sq:
                continue

            logical_col, logical_row = sq             
            visual_col = logical_row + 1
            visual_row = logical_col

            highlight_surface = pygame.Surface((TILE_SIZE - GAP, TILE_SIZE - GAP), pygame.SRCALPHA)
            
            if desc == "availble moves":
                highlight_surface.fill(COLOR_HIGHLIGHT)
            elif desc == "check":
                highlight_surface.fill(COLOR_CHECK)
            elif desc == "desync":
                highlight_surface.fill(COLOR_DESYNC)
            
            self.screen.blit(highlight_surface, (visual_col * TILE_SIZE + GAP // 2, visual_row * TILE_SIZE + GAP // 2))
            
            if desc == "desync" and "figure" in status:
                fig_id = status["figure"]
                text = PIECES_UNICODE["white"].get(fig_id, "?") 
                text_surface = self.font.render(text, True, (255, 0, 0)) 
                text_rect = text_surface.get_rect(center=((visual_col) * TILE_SIZE + TILE_SIZE // 2, 
                                                          visual_col * TILE_SIZE + TILE_SIZE // 2))
                self.screen.blit(text_surface, text_rect)

        # 4. Отрисовка нормальных фигур
        board = state.get("board", [])
        for logical_row in range(ROWS):
            visual_row = 7 - logical_row
            
            for col_8x8 in range(8):
                
                piece_data = board[logical_row][col_8x8]
                if "piece" in piece_data:
                    visual_col = logical_row + 1
                    visual_row = col_8x8

                    p_id = piece_data["piece"]
                    p_color = piece_data["color"]
                    
                    text = PIECES_UNICODE[p_color].get(p_id, "?")
                    text_color = (0, 0, 0) if p_color == "black" else (255, 255, 255)
                    
                    text_surface = self.font.render(text, True, text_color)
                    text_rect = text_surface.get_rect(center=(visual_col * TILE_SIZE + TILE_SIZE // 2, 
                                                              visual_row * TILE_SIZE + TILE_SIZE // 2))
                    self.screen.blit(text_surface, text_rect)
        

        # 6. Экран окончания игры
        game_over_text = None
        for status in statuses:
            desc = status.get("description", "")
            if desc in ["mate", "checkmate"]:
                is_white_turn = state.get("is_white_turn", True)
                winner = "ЧЕРНЫЕ" if is_white_turn else "БЕЛЫЕ"
                game_over_text = f"МАТ! ПОБЕДИЛИ {winner}"
            elif desc in ["stalemate", "draw"]:
                game_over_text = "НИЧЬЯ!"

        if game_over_text:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180)) 
            self.screen.blit(overlay, (0, 0))

            big_font = pygame.font.SysFont("Arial", 60, bold=True)
            text_surf = big_font.render(game_over_text, True, (255, 50, 50))
            text_rect = text_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            
            shadow_surf = big_font.render(game_over_text, True, (0, 0, 0))
            shadow_rect = shadow_surf.get_rect(center=(WIDTH // 2 + 3, HEIGHT // 2 + 3))
            
            self.screen.blit(shadow_surf, shadow_rect)
            self.screen.blit(text_surf, text_rect)
            
# --- ГЛАВНЫЙ ЦИКЛ ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess Client")
    clock = pygame.time.Clock()

    network = NetworkManager()
    renderer = Renderer(screen)
    
    state = network.get_start_state(True)
    last_tyme_sync = time.time()

    running = True
    while running:
        current_time = time.time()

        if state:
            if current_time - last_tyme_sync > 1.0:
                state = network.get_board_state()
                last_tyme_sync = current_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        print(state)
        renderer.draw_board(state)
        pygame.display.flip()
        
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()