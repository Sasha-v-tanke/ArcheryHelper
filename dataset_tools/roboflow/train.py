from ultralytics import YOLO

# Создаем модель (начнем с pretrained YOLOv8n)
model = YOLO("yolov8n.pt")

# Обучаем на скачанном датасете (укажи путь к data.yaml)
model.train(data="archery-1/data.yaml", epochs=5, imgsz=640)
