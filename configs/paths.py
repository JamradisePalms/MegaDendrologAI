from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parent().parent()
ML_PATH = ROOT_PATH / 'ML'
CLASSIFICATION_PATH = ML_PATH / 'Classification'

class PathConfig:
    class ML:
        class Classification:
            prompt_filepath = CLASSIFICATION_PATH / 'classification_prompt_v1.md'
        class Detection:
            