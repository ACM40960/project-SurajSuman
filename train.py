# Activate the yolood environment yolood
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(data="C:/Users/imsum/UCD/Summer/Project/yolo_od/PascalYoloFull/data.yaml",
            epochs=100,patience=10,batch=8,
            lr0=0.00022759097966394, # Best learning rate found from tuning
            imgsz=640,
            momentum=0.8719857796351089, # Best momentum found from tuning
            weight_decay=1.758667459412074e-06, # Best weight decay found from tuning
            workers=0,
            optimizer='AdamW',
            freeze=10,
            cos_lr=True,
            verbose=False)

#model = YOLO("yolov8m.pt")

#model.train(data = "data.yaml", batch=8, imgsz=640, epochs=100, wprkers=1)

# Cmd command:
# yolo task=detect mode=train epochs=100 data=data.yaml model=yolov8m.pt imgsz=640