from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from ML.Classification.vlm_lib.Dataclasses import DetectionTreeAnalysis, ClassificationTreeAnalysis
from ML.Classification.vlm_lib.QwenImageClassifier import QwenImageClassifier
from ML.Classification.vlm_lib.utils import write_json
from configs.paths import PathConfig

CLASSIFICATION_PATHS = PathConfig.ML.Classification
CLASSIFICATION_PROMPT_FILEPATH = CLASSIFICATION_PATHS.CLASSIFICATION_PROMPT_FILEPATH

path_to_source_images = Path('Hack-processed-data/visualization')
path_to_crop_images = Path('Hack-processed-data/tree_crops')

images_dataset = []
for crop_image_filepath in path_to_crop_images.iterdir():
    true_name = crop_image_filepath.name[12:]
    source_image_filepath = path_to_source_images / f'det_{true_name}'
    images_dataset.append((crop_image_filepath, source_image_filepath))

classifier = QwenImageClassifier(CLASSIFICATION_PROMPT_FILEPATH, ClassificationTreeAnalysis)
response = classifier.run(next(iter(images_dataset)))
write_json(response, Path('Hack-processed-data/result.json'))