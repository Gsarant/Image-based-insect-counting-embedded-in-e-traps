import sys
sys.path.append('../../ultralytics')
from ultralytics import YOLO
                
model = YOLO("yolov8n.pt") # pass any model type
model.train(
    data='myyalm/helicoverpa.yaml',
    workers=4,
    batch=32,
    imgsz=480,
    epochs=300,
    pretrained=True,
    name='Yolov8runs_10k/Helicoverpa-armigera'
)
model2 = YOLO("yolov8n.pt") # pass any model type

model2.train(
    data='myyalm/plodia.yaml',
    workers=4,
    batch=32,
    imgsz=480,
    epochs=300,
    pretrained=True,
    name='Yolov8runs_10k/Plodia'
)