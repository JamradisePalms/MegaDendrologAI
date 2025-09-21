from pathlib import Path
import os

ROOT_PATH: Path = Path(__file__).resolve().parent().parent()
ML_DIR = ROOT_PATH / 'ML'
CLASSIFICATION_DIR = ML_DIR / 'Classification'
DETECTION_DIR = ML_DIR / 'Detection'
DETECTION_DATASET_DIR = DETECTION_DIR / 'Data'

paths = [
    ROOT_PATH,
    ML_DIR,
    CLASSIFICATION_DIR,
    DETECTION_DIR,
    DETECTION_DATASET_DIR
]

for path in paths:
    path.mkdir(parents=True, exist_ok=True)

class PathConfig:
    class ML:
        class Classification:
            PROMPT_FILEPATH = CLASSIFICATION_DIR / 'classification_prompt_v1.md'
        class Detection:
            PATH_TO_SAVE_DATASET = DETECTION_DATASET_DIR
            