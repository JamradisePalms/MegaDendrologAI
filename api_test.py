from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from ML.Classification.vlm_lib.Dataclasses import ClassificationTreeAnalysis
from ML.Classification.vlm_lib.QwenImageClassifier import QwenImageClassifier
from ML.Classification.vlm_lib.utils import write_json
from configs.paths import PathConfig

CLASSIFICATION_PATHS = PathConfig.ML.Classification
CLASSIFICATION_PROMPT_FILEPATH = CLASSIFICATION_PATHS.CLASSIFICATION_PROMPT_FILEPATH
PATH_TO_FULL_DATASET = Path(r"C:\Users\shari\OneDrive\Рабочий стол\Hack-processed-data\tree_crops/")

classifier = QwenImageClassifier(CLASSIFICATION_PROMPT_FILEPATH, ClassificationTreeAnalysis)
response = classifier.run(PATH_TO_FULL_DATASET.iterdir(), max_workers=3)
write_json(response, Path('result.json'))