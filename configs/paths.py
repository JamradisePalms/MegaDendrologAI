from pathlib import Path
import os

ROOT_PATH: Path = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_PATH / "ML"
WEB_DATASET_DIR = ML_DIR / 'web_data'
CLASSIFICATION_DIR = ML_DIR / "Classification"
DETECTION_DIR = ML_DIR / "Detection"
DETECTION_DATASET_DIR = DETECTION_DIR / "Data"
YOLO_RUNS_DIR = DETECTION_DIR / "yolo_train"
PATH_TO_YOLO_CONFIG_YAML = YOLO_RUNS_DIR / "yolo_config.yaml"
RESULT_DIR = ML_DIR / "results"
CLASSIFICATION_PROMPTS_DIR = CLASSIFICATION_DIR / "prompts"

paths = [
    ROOT_PATH,
    ML_DIR,
    CLASSIFICATION_DIR,
    DETECTION_DIR,
    DETECTION_DATASET_DIR,
    YOLO_RUNS_DIR,
    RESULT_DIR,
    CLASSIFICATION_PROMPTS_DIR,
    WEB_DATASET_DIR,
]

for path in paths:
    path.mkdir(parents=True, exist_ok=True)


class PathConfig:
    class ML:
        class Classification:
            CLASSIFICATION_PROMPT_FILEPATH = (
                CLASSIFICATION_PROMPTS_DIR / "classification_prompt_v1.md"
            )
            TREE_TYPE_PROMPT_FILEPATH = CLASSIFICATION_PROMPTS_DIR / "tree_type.md"
            PATH_TO_DEBUG_IMAGES = CLASSIFICATION_DIR / "debug_images"
            PATH_TO_IMAGES = ML_DIR / "images"
            TREE_TYPE_DATASET_FILEPATH = RESULT_DIR / "tree_types.json"
            CLASSIFICATION_DATASET_FILEPATH = RESULT_DIR / "classification_result.json"

        class Detection:
            PATH_TO_SAVE_DATASET = DETECTION_DATASET_DIR
            PATH_TO_SAVE_TRAIN_RUNS = YOLO_RUNS_DIR
            PATH_TO_YOLO_CONFIG = PATH_TO_YOLO_CONFIG_YAML

            PATH_TO_YOLO_SMALL_BEST_WEIGHTS = Path("/home/jamradise/MegaDendrologAI/ML/Detection/yolo_weights/yolov11s/best.pt")
            PATH_TO_YOLO_SMALL_LAST_WEIGHTS = Path("/home/jamradise/MegaDendrologAI/ML/Detection/yolo_weights/yolov11s/last.pt")

            PATH_TO_YOLO_MEDIUM_BEST_WEIGHTS = Path("/home/jamradise/MegaDendrologAI/ML/Detection/yolo_weights/yolov11m/best.pt")
            PATH_TO_YOLO_MEDIUM_LAST_WEIGHTS = Path("/home/jamradise/MegaDendrologAI/ML/Detection/yolo_weights/yolov11s/last.pt")

            PATH_TO_YOLO_LARGE_BEST_WEIGHTS = Path("/home/jamradise/MegaDendrologAI/ML/Detection/yolo_weights/yolov11l/best.pt")
            PATH_TO_YOLO_LARGE_BEST_WEIGHTS = Path("/home/jamradise/MegaDendrologAI/ML/Detection/yolo_weights/yolov11l/best.pt")

        PATH_TO_RAW_IMAGES = DETECTION_DATASET_DIR / "Hack-raw-data"
        PATH_TO_SAVE_PROCESSED_IMAGES = DETECTION_DATASET_DIR / "Hack-processed-data"
        PATH_TO_FULL_DATASET = ROOT_PATH / 'dataset_to_mark'
