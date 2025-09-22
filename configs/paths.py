from pathlib import Path
import os

ROOT_PATH: Path = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_PATH / 'ML'
CLASSIFICATION_DIR = ML_DIR / 'Classification'
DETECTION_DIR = ML_DIR / 'Detection'
DETECTION_DATASET_DIR = DETECTION_DIR / 'Data'
YOLO_RUNS_DIR = DETECTION_DIR / 'yolo_train'
PATH_TO_YOLO_CONFIG_YAML = YOLO_RUNS_DIR / 'yolo_config.yaml'

paths = [
    ROOT_PATH,
    ML_DIR,
    CLASSIFICATION_DIR,
    DETECTION_DIR,
    DETECTION_DATASET_DIR,
    YOLO_RUNS_DIR,
]

for path in paths:
    path.mkdir(parents=True, exist_ok=True)

class PathConfig:
    class ML:
        class Classification:
            PROMPT_FILEPATH = CLASSIFICATION_DIR / 'classification_prompt_v1.md'
        class Detection:
            PATH_TO_SAVE_DATASET = DETECTION_DATASET_DIR
            PATH_TO_SAVE_TRAIN_RUNS = YOLO_RUNS_DIR
            PATH_TO_YOLO_CONFIG = PATH_TO_YOLO_CONFIG_YAML
            