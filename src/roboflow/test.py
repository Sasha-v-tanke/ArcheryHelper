from ultralytics import YOLO
import cv2

from path_manager import BASE_DIR

# Загружаем модель
model = YOLO(BASE_DIR + "/runs/detect/train6/weights/best.pt")

# Загружаем изображение
img = "/Users/alex/Projects/Python/archery/data/normalized-new/10.jpeg"

# Делаем предсказание
results = model(img)
boxes = results[0].boxes.xyxy.cpu().numpy()
print(boxes)

# Показ результатов
# results.show()  # откроет изображение с детекцией
# results.save("runs/detect/predictions")  # сохранит результат
