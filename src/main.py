import cv2
from fastapi import FastAPI, Response

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

@app.get("/calibrate")
async def calibrate_camera():
    cap = cv2.VideoCapture(0)
    _, image = cap.read()
    cv2.imwrite("calibration.jpg", image)
    cap.release()

    app = FastAPI(title="Stand api")

    # image = cv2.imread('./real_caklib.png')
    if not pipeline.process_image(image):
        raise RuntimeError("couldn't detect calibration")
    return {
        "rows": pipeline._cols,
        "cols": pipeline._rows
    }

@app.get("/data")
async def process_step():
    cap = cv2.VideoCapture(0)
    import time
    time.sleep(1)

    _, image = cap.read()
    
    # image = cv2.imread('./real_caklib.png')
    cap.release()
    return pipeline.get_json_data(image)

@app.get("/image")
async def process_step():
    cap = cv2.VideoCapture(0)
    import time
    time.sleep(1)

    _, image = cap.read()
    
    cap.release()
    res, im_png = cv2.imencode(".png", image)

    return Response(content=im_png.tobytes(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)