# Activate the yolood environment yolood
from ultralytics import YOLO

model = YOLO("./runs/detect/train7/weights/best.pt")

model.predict(source=0, show=True, save=True, conf=0.5)

# cmd command:
# yolo task=detect mode=predict model=.\runs\detect\train3\weights\best.pt show=True conf=0.5 source=0