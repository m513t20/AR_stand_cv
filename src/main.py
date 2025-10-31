import cv2

from Detection import CalibrationPipeline

if __name__ == "__main__":
    pipeline = CalibrationPipeline()
    cap = cv2.VideoCapture(0)
    _,image = cap.read()
    # image = cv2.imread('./template_cut.png')

    if pipeline.process_image(image):
        _,image = cap.read()
        out = pipeline.get_board_data(image)
        print(out)