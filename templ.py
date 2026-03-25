import cv2

# 1. Инициализация камеры (0 - индекс стандартной камеры)
cap = cv2.VideoCapture(0)

# 2. Настройка словаря и параметров детектора
# Используем именно DICT_6X6_250, как вы и просили
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

print("Нажмите 'q' для выхода")

while True:
    # Захват кадра
    ret, frame = cap.read()
    if not ret:
        break

    # Обнаружение маркеров
    # (Детектор сам конвертирует кадр, если нужно, но серое изображение работает быстрее)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    # Если маркеры найдены, рисуем их
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Пример вывода ID в консоль
        for i in range(len(ids)):
            print(f"Обнаружен маркер с ID: {ids[i][0]}")

    # Показ кадра
    cv2.imshow('ArUco Recognition', frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
