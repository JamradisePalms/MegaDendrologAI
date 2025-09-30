from tkinter.constants import Y
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from ML.Classification.vlm_lib.Dataclasses import ClassificationTreeAnalysis
from ML.Classification.vlm_lib.YndxImageClassifier import YndxImageClassifier
from ML.Classification.vlm_lib.utils import write_json
from configs.paths import PathConfig

CLASSIFICATION_PATHS = PathConfig.ML.Classification
CLASSIFICATION_PROMPT_FILEPATH = CLASSIFICATION_PATHS.CLASSIFICATION_PROMPT_FILEPATH
PATH_TO_FULL_DATASET = Path("/Users/pasheeee/MyProjects/MegaDendrologAI/tree_crops")

classifier = YndxImageClassifier(CLASSIFICATION_PROMPT_FILEPATH, ClassificationTreeAnalysis)
images = list(PATH_TO_FULL_DATASET.iterdir())
response = classifier.run(images, max_workers=10)
write_json(response, Path('result.json'))