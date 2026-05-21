 # --- Настройки Сети ---
BROKER_ADDRESS = "localhost"  # IP, где поднят docker-compose
BROKER_PORT = 1883
API_BASE_URL = "http://localhost:8000" # URL FastAPI бэкенда

# --- Идентификация Стенда ---
STAND_ID = "hw_stand_001"
PIN_CODE = "1234"  # Пин-код для привязки в Dashboard лобби

# --- Настройки Pygame ---
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 640
FPS = 30
SQUARE_SIZE = SCREEN_WIDTH // 8