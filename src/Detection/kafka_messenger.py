import json
from typing import List

import cv2
import numpy as np
from kafka import KafkaProducer

from Detection.pipeline import CalibrationPipeline

class KafkaMessenger:
    
    def __init__(self, servers: List[str], image: np.ndarray, topic: str = "stand/cv"):
        self._producer = KafkaProducer(
            bootstrap_servers = servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8') 
        )
        self._pipeline = CalibrationPipeline()
        self._topic = topic

        if not self._pipeline.process_image(image):
            raise RuntimeError("couldn't detect calibration")
        
    def send_data(self, image: np.ndarray):
        data = self._pipeline.get_json_data(image)
        # print(data)
        self._producer.send(self._topic, data)
        
        self._producer.flush()
        self._producer.close()