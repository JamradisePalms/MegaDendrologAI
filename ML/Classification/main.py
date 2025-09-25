from ML.Classification.vlm_lib.QwenImageClassifier import QwenImageClassifier
from configs.paths import PathConfig
from ML.Classification.vlm_lib.utils import write_json

CLASSIFICATION_PATHS = PathConfig.ML.Classification

CLASSIFICATION_PROMPT_FILEPATH = CLASSIFICATION_PATHS.CLASSIFICATION_PROMPT_FILEPATH
PATH_TO_IMAGES = CLASSIFICATION_PATHS.PATH_TO_IMAGES

classifier = QwenImageClassifier(CLASSIFICATION_PROMPT_FILEPATH)
response = classifier.run(PATH_TO_IMAGES)
write_json(
    response,
)
