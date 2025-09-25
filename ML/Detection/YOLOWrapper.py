import ultralytics
import yaml
from pathlib import Path
import torch
from datetime import datetime
from configs.paths import PathConfig
from ML.Detection.utils import create_default_config
from ML.logging_config import setup_logging
from BaseDetection import BaseDetection
import logging

logger = logging.getLogger(__name__)

PATH_TO_SAVE_TRAIN_RUNS = PathConfig.ML.Detection.PATH_TO_SAVE_TRAIN_RUNS
PATH_TO_YOLO_CONFIG_YAML = PathConfig.ML.Detection.PATH_TO_YOLO_CONFIG

class YoloWrapper(BaseDetection):
    def __init__(self, model_version: str = '11', model_size: str = 'n', finetune: bool = True, device=None, weights_path: str = None, config_path: str = None):
        """
        model_version: версия YOLO (например, '11')
        model_size: размер модели ('n', 's', 'm', 'l', 'x')
        finetune: True - использовать .pt для дообучения, False - .yaml для обучения с нуля
        device: GPU id (0,1,..) или 'cpu', None - автоматический выбор
        weights_path: путь к зафайнтюненной модели для загрузки
        """
        super().__init__(weights_path=weights_path, config_path=config_path, device=device)

        if weights_path:
            logger.info("Loading weights for YOLO from %r", weights_path)

            self._model = ultralytics.YOLO(weights_path)
        else:
            if finetune:
                model_file = "yolo" + f"{model_version}{model_size}.pt"
                logger.info("Loading trained %s on COCO dataset", model_file)
            else:
                model_file = f"{model_version}{model_size}.yaml"
                logger.info("Loading random initialized %s", model_file)

            self._model = ultralytics.YOLO(model_file)
    
    @property
    def model(self):
        return self._model

    def predict(self, image_path: Path, save=False, save_dir="runs/predict", **kwargs):
        predict_kwargs = {**self.config, **kwargs}

        try:
            results = self.model.predict(
                source=str(image_path),
                **predict_kwargs
            )
        except Exception:
            logger.exception(f"Failed to process data from {image_path}", exc_info=True)
            return None
        return results

if __name__ == "__main__":
    setup_logging()
    create_default_config()
    