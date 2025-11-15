import cv2

from Detection import KafkaMessenger

if __name__ == "__main__":
    # on stand
    # cap = cv2.VideoCapture(0)
    # _,image = cap.read()
    
    # debug
    image = cv2.imread('./real_caklib.png')

    # messenger
    messenger = KafkaMessenger(['localhost:8000'], image)

    while True:
        # on stand
        # _,image = cap.read()
        messenger.send_data(image)
