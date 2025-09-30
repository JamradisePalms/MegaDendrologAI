from pathlib import Path
from transformers import AutoImageProcessor


class TrainConfigs:
    class HollowClassification:
        MODEL_NAME = 'microsoft/resnet-50'
        TRAIN_JSON_FILEPATH = Path("Hack-processed-data/result.json")
        IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained(
            MODEL_NAME, use_fast=True
        )
        IMAGE_JSON_FIELD = "image"
        TARGET_JSON_FIELD = "has_hollow"
        BATCH_SIZE = 200
        NUM_EPOCHS = 100
        NUM_LABELS = 2
        LR = 5e-5
        PATH_TO_SAVE_MODEL = Path('ML/Classification/results/saved_models/hollow_classification_v1.pth')
    
    class HollowClassificationSmallModel:
        MODEL_NAME = 'microsoft/resnet-18'
        TRAIN_JSON_FILEPATH = Path("Hack-processed-data/result.json")
        IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained(
            MODEL_NAME, use_fast=True
        )
        IMAGE_JSON_FIELD = "image"
        TARGET_JSON_FIELD = "has_hollow"
        BATCH_SIZE = 700
        NUM_EPOCHS = 100
        NUM_LABELS = 2
        LR = 5e-5
        PATH_TO_SAVE_MODEL = Path('ML/Classification/results/saved_models/hollow_classification_small.pth')

    class TreeClassificationModelWithMultiHeadMLP:
        MODEL_NAME = 'microsoft/resnet-18'
        TRAIN_JSON_FILEPATH = Path("result.json")
        IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained(
            MODEL_NAME, use_fast=True
        )
        IMAGE_JSON_FIELD = "image"
        TASK_TO_OUTPUT_SIZE = {
            "has_hollow": 2,
            "has_cracks": 2,
            "has_fruits_or_flowers": 2,
            'overall_condition': 5,
            'has_rot': 2,
            'has_trunk_damage': 2,
            'has_crown_damage': 2,
            'dry_branch_percentage': 6,
        }

        BATCH_SIZE = 20
        NUM_EPOCHS = 20
        LR = 1e-4
        PATH_TO_SAVE_MODEL = Path('ML/Classification/results/saved_models/hollow_classification_small.pth')