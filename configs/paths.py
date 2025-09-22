from pathlib import Path
import os

ROOT_PATH: Path = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_PATH / 'ML'
CLASSIFICATION_DIR = ML_DIR / 'Classification'
DETECTION_DIR = ML_DIR / 'Detection'
DETECTION_DATASET_DIR = DETECTION_DIR / 'Data'
YOLO_RUNS_DIR = DETECTION_DIR / 'yolo_train'
PATH_TO_YOLO_CONFIG_YAML = YOLO_RUNS_DIR / 'yolo_config.yaml'
CLASSIFICATION_RESULT_DIR = CLASSIFICATION_DIR / 'result'

paths = [
    ROOT_PATH,
    ML_DIR,
    CLASSIFICATION_DIR,
    DETECTION_DIR,
    DETECTION_DATASET_DIR,
    YOLO_RUNS_DIR,
    CLASSIFICATION_RESULT_DIR
]

for path in paths:
    path.mkdir(parents=True, exist_ok=True)

class PathConfig:
    class ML:
        class Classification:
            CLASSIFICATION_PROMPT_FILEPATH = CLASSIFICATION_DIR / 'classification_prompt_v1.md'
            TREE_TYPE_PROMPT_FILEPATH = CLASSIFICATION_DIR / 'tree_type.md'
            PATH_TO_TEST_IMAGES = CLASSIFICATION_DIR / 'images'
            TREE_TYPE_DATASET_FILEPATH = CLASSIFICATION_RESULT_DIR / 'tree_types.json'
            CLASSIFICATION_DATASET_FILEPATH = CLASSIFICATION_RESULT_DIR / 'classification_result.json'
        class Detection:
            PATH_TO_SAVE_DATASET = DETECTION_DATASET_DIR
            PATH_TO_SAVE_TRAIN_RUNS = YOLO_RUNS_DIR
            PATH_TO_YOLO_CONFIG = PATH_TO_YOLO_CONFIG_YAML
            PATH_TO_SAVE_PROCESSED_IMAGES = DETECTION_DATASET_DIR / "ProcessedData"
        
        PATH_TO_RAW_IMAGES = ML_DIR / "Data"