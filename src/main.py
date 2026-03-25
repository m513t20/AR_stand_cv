import cv2
from fastapi import FastAPI, HTTPException

from Detection import CalibrationPipeline

pipeline = CalibrationPipeline()

cap = cv2.VideoCapture(0)
_, image = cap.read()
cv2.imwrite("calibration.jpg", image)
cap.release()

app = FastAPI(title="Stand api")

# image = cv2.imread('./real_caklib.png')
if not pipeline.process_image(image):
    raise RuntimeError("couldn't detect calibration")

print(f'calibration finished {pipeline._rows}x{pipeline._cols}')

@app.get("/data")
async def process_step():
    cap = cv2.VideoCapture(0)
    import time
    time.sleep(1)

    _, image = cap.read()
    
    # image = cv2.imread('./real_caklib.png')
    cap.release()
    return pipeline.get_json_data(image)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)