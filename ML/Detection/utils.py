import yaml
from configs.paths import PathConfig
import os

PATH_TO_YOLO_CONFIG_YAML = PathConfig.ML.Detection.PATH_TO_YOLO_CONFIG
YOLO_RUNS_DIR = PathConfig.ML.Detection.PATH_TO_SAVE_TRAIN_RUNS / "runs/train"
PATH_TO_DATA = PathConfig.ML.Detection.PATH_TO_SAVE_DATASET / "Tree-Quality-3-7/data.yaml"

def create_default_config(path=PATH_TO_YOLO_CONFIG_YAML):
        """
        Создаёт YAML конфиг с базовыми параметрами обучения.
        """
        default_config = {
            "data": str(PATH_TO_DATA),
            "epochs": 300,
            "imgsz": 640,
            "batch": 16,
            "device": None,
            "workers": 8,
            "optimizer": "SGD",
            "lr0": 0.005,
            "lrf": 0.001,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "patience": 30,
            "resume": False,
            "freeze": 0,
            "pretrained": True,
            "save": True,
            "save_period": -1,
            "verbose": True,
            "plots": True,
            "project": str(YOLO_RUNS_DIR),
            "name": None,
            "exist_ok": False,
            "cache": False,
            "rect": False,
            "cos_lr": False,
            "close_mosaic": 10,
            "amp": True
        }

        if os.path.exists(path):
            print(f"Config is already exists {path}")
        else:
            with open(path, "w") as f:
                yaml.dump(default_config, f)
            print(f"Default config saved to {path}")