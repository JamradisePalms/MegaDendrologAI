from pathlib import Path
from transformers import AutoImageProcessor
import torchvision.transforms as transforms
from ML.Classification.torch_lib.ImagePreprocessor import TreeImagePreprocessor

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

    class TreeClassificationModelWithMultiHeadMLP():
        MODEL_NAME = 'efficientnet-b1'
        BACKBONE_TYPE = 'efficientnet'
        TRAIN_JSON_FILEPATH = Path("train_data.json")
        VAL_JSON_FILEPATH = Path("valid_data.json")
        METRIC = "task_losses"

        PATIENCE = 10
        MIN_DELTA = 0.001
        
        IMAGE_JSON_FIELD = "image"
        TARGET_JSON_FIELD = {
            "tree_type": 27,
            "has_hollow": 2,
            "has_cracks": 2,
            "has_fruits_or_flowers": 2,
            "overall_condition": 6,
            "has_crown_damage": 2,
            "has_trunk_damage": 2,
            "has_rot": 2
        }

        LOSS_WEIGHTS = {
            "tree_type": 1.5,
            "has_hollow": 0.3,
            "has_cracks": 0.7,
            "has_fruits_or_flowers": 0.2,
            "overall_condition": 1.0,
            "has_crown_damage": 0.4,
            "has_trunk_damage": 0.8,
            "has_rot": 0.15
        }

        BATCH_SIZE = 16
        NUM_EPOCHS = 60
        LR = 1e-4
        PATH_TO_SAVE_MODEL = Path('ML/Classification/results/saved_models/multi_head_tree_classification.pth')

        @classmethod
        def get_image_processor(cls):
            return cls.get_image_preprocessor(is_train=False)

        @classmethod
        def get_image_preprocessor(cls, is_train: bool = False):
            return TreeImagePreprocessor(
                model_name=cls.MODEL_NAME,
                backbone_type=cls.BACKBONE_TYPE,
                image_size=224,
                is_train=is_train
            )