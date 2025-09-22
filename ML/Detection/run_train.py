import ultralytics
import yaml
from pathlib import Path
import torch
from datetime import datetime
from configs.paths import PathConfig
from ML.Detection.utils import create_default_config
from ML.Detection.YOLOWrapper import YOLOWrapper

PATH_TO_SAVE_TRAIN_RUNS = PathConfig.ML.Detection.PATH_TO_SAVE_TRAIN_RUNS
PATH_TO_YOLO_CONFIG_YAML = PathConfig.ML.Detection.PATH_TO_YOLO_CONFIG


create_default_config()

wrapper = YOLOWrapper(model_version="v11", model_size="n", finetune=True)
wrapper.train(config_path=PATH_TO_YOLO_CONFIG_YAML, batch=32)

# finetuned = YOLOWrapper(weights_path="runs/train/20250921_123456/weights/best.pt")

# results = wrapper.predict(r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Detection\data", conf=0.3, save=True)