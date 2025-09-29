from ultralytics import YOLO
from configs.paths import PathConfig

path_to_yolo_weights = PathConfig.ML.Detection.PATH_TO_BEST_YOLO_WEIGHTS

model = YOLO(path_to_yolo_weights)

model.export(format='tflite', int8=False, dynamic=True)
