import cv2

from Detection import CalibrationPipeline

if __name__ == "__main__":
    pipeline = CalibrationPipeline()
    if pipeline.process_image('./template_cut.png'):
        image = cv2.imread('./template_cut.png')
        out = pipeline.get_board_data(image)
        print(out)