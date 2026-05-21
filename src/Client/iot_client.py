# iot_client.py
import paho.mqtt.client as mqtt
import json
import time
import threading
import cv2

from Detection import CalibrationPipeline

class StandClient:
    def __init__(self, config, on_board_update):
        self.config = config
        self.client = mqtt.Client(client_id=self.config.STAND_ID)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        self.on_board_update = on_board_update
        self.active_game_id = None
        self.is_playing = False
        
        # калиборуемся камерой и выводим результат
        self.pipeline = CalibrationPipeline()
        self.cap = cv2.VideoCapture(0)
        _, image = self.cap.read()
        # image = cv2.imread("real_caklib.png")
        cv2.imwrite("calibration.jpg", image)
        if not self.pipeline.process_image(image):
            raise RuntimeError("couldn't detect calibration")
        print(f'calibration finished {self.pipeline._rows}x{self.pipeline._cols}')


    def connect(self):
        self.client.connect(self.config.BROKER_ADDRESS, self.config.BROKER_PORT, 60)
        self.client.loop_start()
        
    def on_connect(self, client, userdata, flags, rc):
        print(f"MQTT Подключен! Слушаем команды для стенда {self.config.STAND_ID}...")
        # 1. Подписка на команды от Lobby Orchestrator
        self.client.subscribe(f"stands/{self.config.STAND_ID}/commands")
        
    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        
        # Слушаем команды управления стендом
        if topic == f"stands/{self.config.STAND_ID}/commands":
            data = json.loads(payload)
            
            if data.get("action") == "start_game":
                self.active_game_id = data.get("game_id")
                role = data.get("role")
                print(f"Получена команда start_game! Game ID: {self.active_game_id}, Роль: {role}")
                
                # 2. Подписываемся на состояние конкретной игры
                self.client.subscribe(f"game/{self.active_game_id}/state")
                
                # Запускаем отправку данных от камеры в отдельном потоке
                self.is_playing = True
                threading.Thread(target=self.send_markers_loop, daemon=True).start()
                
        # Слушаем состояние игры (от Game Engine)
        elif self.active_game_id and topic == f"game/{self.active_game_id}/state":
            try:
                data = json.loads(payload)
                if "fen" in data:
                    self.on_board_update(data["fen"])
            except Exception as e:
                print(f"Ошибка парсинга FEN: {e}")

    def send_markers_loop(self):
        """Симулирует отправку данных с камеры / Aruco маркеров"""
        topic = f"game/{self.active_game_id}/{self.config.STAND_ID}/markers"
        
        while self.is_playing:
            # Заглушка: тут ты будешь отправлять реальные кординаты маркеров с камеры.
            # По документации: 1-6 это белые фигуры, 11-16 - черные.
            ret, frame = self.cap.read()
            # frame = cv2.imread("real_caklib.png")
            data = self.pipeline.get_json_data(frame)
            payload = json.loads(data)
            self.client.publish(topic, json.dumps(payload))
            time.sleep(2)  # Частота кадров (FPS) с камеры