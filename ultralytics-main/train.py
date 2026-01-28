import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/root/code/dataset/dataset_visdrone/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=8, 
                optimizer='SGD', # using SGD
                # device='0,1', 
                # patience=0, # set 0 to close earlystop.
                # resume=True, 
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )