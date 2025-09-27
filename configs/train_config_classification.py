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