# chess_gui.py
import pygame
import sys

class ChessRenderer:
    def __init__(self, config):
        pygame.init()
        self.config = config
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption(f"AR Chess Client - {config.STAND_ID}")
        self.clock = pygame.time.Clock()
        
        fonts = 'segoeuisymbol,dejavusans,freeserif,arial'
        self.font = pygame.font.SysFont(fonts, int(config.SQUARE_SIZE * 0.7))
        
        # Стартовая позиция (стандартная FEN)
        self.board_state = self._fen_to_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        
        self.pieces_unicode = {
            'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
            'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙'
        }

    def update_board_from_fen(self, fen):
        """Парсит FEN от бэкенда и обновляет доску"""
        self.board_state = self._fen_to_board(fen)

    def _fen_to_board(self, fen):
        """Вспомогательная функция: FEN -> 2D Массив 8x8"""
        board = []
        rows = fen.split(' ')[0].split('/')
        for row in rows:
            board_row = []
            for char in row:
                if char.isdigit():
                    board_row.extend(['.'] * int(char))
                else:
                    board_row.append(char)
            board.append(board_row)
        return board

    def draw_board(self):
        colors = [(238, 238, 210), (118, 150, 86)]
        sq_sz = self.config.SQUARE_SIZE
        
        for row in range(8):
            for col in range(8):
                pygame.draw.rect(self.screen, colors[(row + col) % 2], 
                                 pygame.Rect(col*sq_sz, row*sq_sz, sq_sz, sq_sz))
                piece = self.board_state[row][col]
                if piece != '.':
                    char = self.pieces_unicode.get(piece, piece)
                    text_surface = self.font.render(char, True, (0, 0, 0))
                    text_rect = text_surface.get_rect(center=(col*sq_sz + sq_sz//2, row*sq_sz + sq_sz//2))
                    self.screen.blit(text_surface, text_rect)

    def render_loop(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        self.draw_board()
        pygame.display.flip()
        self.clock.tick(self.config.FPS)