from pathlib import Path
from ML.Classification.torch_lib.ImagePreprocessor import TreeImagePreprocessor
from transformers import AutoImageProcessor

FREEZE_DICT = {
    # DeiT
    "deit_tiny_patch16_224": 6,      # всего 12 блоков, заморозим половину
    "deit_small_patch16_224": 8,     # всего 12 блоков, заморозим 2/3
    "deit_base_patch16_224": 10,     # всего 12 блоков, заморозим почти все

    "swin_tiny_patch4_window7_224": 2,   # 4 стадии, заморозим первые 2 стадии
    "swin_small_patch4_window7_224": 3,  # 4 стадии, заморозим первые 3

    "mobilevit_xxs": 4,  # 4 блока
    "mobilevit_xs": 6,   # чуть больше блоков
    "mobilevit_s": 8,    # большие модели — меньше фриза

    "efficientformer_l1": 4,   # первые 4 слоя заморозить
    "efficientformer_l3": 6,   # первые 6 слоев заморозить
    "efficientformer_l7": 8,   # первые 8 слоев заморозить

    "poolformer_s12": 6,
    "poolformer_s24": 12,
    "poolformer_s36": 18,
    "poolformer_m36": 18,
    "poolformer_m48": 24,

    "tiny_vit_5m_224": 4,
    "tiny_vit_11m_224": 6,
    "tiny_vit_21m_224": 8,

    "levit_128s": 4,
    "levit_256": 6,
}

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
        TRAIN_JSON_FILEPATH = Path(r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Classification\new_data\train_data.json")
        VAL_JSON_FILEPATH = Path(r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Classification\new_data\valid_data.json")

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
        MODEL_NAME = 'microsoft/resnet-18'
        BACKBONE_TYPE = 'resnet'
        TRAIN_JSON_FILEPATH = Path(r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Classification\new_data\train_data.json")
        VAL_JSON_FILEPATH = Path(r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Classification\new_data\valid_data.json")
        METRIC = "task_losses"

        PATIENCE = 15
        MIN_DELTA = 0.001
        
        IMAGE_JSON_FIELD = "image"
        TARGET_JSON_FIELD = {
            "tree_type": 27,
            # "has_hollow": 2,
            # "has_cracks": 2,
            # "has_fruits_or_flowers": 2,
            # "overall_condition": 6,
            # "has_crown_damage": 2,
            # "has_trunk_damage": 2,
            # "has_rot": 2
        }

        LOSS_WEIGHTS = {
            # "tree_type": 1.5,
            # "has_hollow": 0.3,
            # "has_cracks": 0.7,
            # "has_fruits_or_flowers": 0.2,
            # "overall_condition": 1.0,
            # "has_crown_damage": 0.4,
            # "has_trunk_damage": 0.8,
            # "has_rot": 0.15
        }

        BATCH_SIZE = 32
        NUM_EPOCHS = 100
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

    class TreeClassificationWithMobileTransformer:
        MODEL_NAME = 'mobilevit_xs'  # mobilevit_xs, efficientformer_l1, poolformer_s12
        BACKBONE_TYPE = 'mobile_transformer'
        TRAIN_JSON_FILEPATH = Path(r"C:\Users\shari\PycharmProjects\MegaDendrologAI\NEW_DATA_TRAIN.json")
        VAL_JSON_FILEPATH = Path(r"C:\Users\shari\PycharmProjects\MegaDendrologAI\NEW_DATA_VAL.json")
        METRIC = "task_losses"
        
        IMAGE_JSON_FIELD = "image"
        TARGET_JSON_FIELD = {
            "tree_type": 24,
            # "has_hollow": 2,
            # "has_cracks": 2,
            # "has_fruits_or_flowers": 2,
            # "overall_condition": 6,
            # # "dry_branch_percentage": 4,
            # "has_crown_damage": 2,
            # "has_trunk_damage": 2,
            # "has_rot": 2
        }

        LOSS_WEIGHTS = {
            "has_hollow": 0.3,
            "has_cracks": 0.7,
            "has_fruits_or_flowers": 0.2,
            "overall_condition": 1.0,
            "dry_branch_percentage": 1.0,
            "has_crown_damage": 0.4,
            "has_trunk_damage": 0.8,
            "has_rot": 0.15
        }

        BATCH_SIZE = 10
        NUM_EPOCHS = 100
        LR = 1e-4
        
        PATIENCE = 20
        MIN_DELTA = 0.002
        
        PATH_TO_SAVE_MODEL = Path('ML/Classification/results/saved_models/APPLE_XS_TRANSFORMER_TREE_TYPE_WEB_DATA.pth')

        @classmethod
        def get_image_processor(cls):
            return cls.get_image_preprocessor(is_train=False)

        @classmethod
        def get_image_preprocessor(cls, is_train: bool = False):
            return TreeImagePreprocessor(
                model_name=cls.MODEL_NAME,
                backbone_type=cls.BACKBONE_TYPE,
                image_size=320,
                is_train=is_train
            )

    class TreeClassificationWithNewTransformers:
        MODEL_NAME = 'deit_tiny_patch16_224' # swin_tiny_patch4_window7_224 or deit_small_patch16_224 or deit_tiny_patch16_224
        BACKBONE_TYPE = 'deit'
        TRAIN_JSON_FILEPATH = Path(r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Classification\new_data\train_data.json")
        VAL_JSON_FILEPATH = Path(r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Classification\new_data\valid_data.json")
        
        METRIC = "task_losses"
        
        IMAGE_JSON_FIELD = "image"
        TARGET_JSON_FIELD = {
            "tree_type": 27,
        }

        LOSS_WEIGHTS = {
        }

        BATCH_SIZE = 8
        NUM_EPOCHS = 150
        LR = 1e-4
        
        PATIENCE = 30
        MIN_DELTA = 0.002
        
        PATH_TO_SAVE_MODEL = Path('ML/Classification/results/saved_models/augs_tiny_deit_trees_27class.pth')

        @classmethod
        def get_image_processor(cls):
            return cls.get_image_preprocessor(is_train=False)

        @classmethod
        def get_image_preprocessor(cls, is_train: bool = False):
            return TreeImagePreprocessor(
                model_name=cls.MODEL_NAME,
                backbone_type=cls.BACKBONE_TYPE,
                image_size=640,
                is_train=is_train
            )