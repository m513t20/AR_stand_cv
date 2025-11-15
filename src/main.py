import cv2

from Detection import CalibrationPipeline

if __name__ == "__main__":
    pipeline = CalibrationPipeline()
    # cap = cv2.VideoCapture(0)
    # _,image = cap.read()
    image = cv2.imread('./real_caklib.png')

    if not pipeline.process_image(image):
        raise RuntimeError("couldn't detect calibration")
    
    # _,image = cap.read()
    out = pipeline.get_json_data(image)