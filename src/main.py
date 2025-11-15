import cv2

from Detection import KafkaMessenger

if __name__ == "__main__":
    # cap = cv2.VideoCapture(0)
    image = cv2.imread('./real_caklib.png')
    messenger = KafkaMessenger(['localhost:8000'], image)

    while True:
        # _,image = cap.read()
        messenger.send_data(image)
