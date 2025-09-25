import ultralytics
import yaml
from pathlib import Path
import torch
from datetime import datetime
from configs.paths import PathConfig
from ML.Detection.utils import create_default_config
from ML.logging_config import setup_logging
import logging

logger = logging.getLogger(__name__)

PATH_TO_SAVE_TRAIN_RUNS = PathConfig.ML.Detection.PATH_TO_SAVE_TRAIN_RUNS
PATH_TO_YOLO_CONFIG_YAML = PathConfig.ML.Detection.PATH_TO_YOLO_CONFIG

class YoloWrapper:
    def __init__(self, model_version: str = '11', model_size: str = 'n', finetune: bool = True, device=None, weights_path: str = None):
        """
        model_version: версия YOLO (например, '11')
        model_size: размер модели ('n', 's', 'm', 'l', 'x')
        finetune: True - использовать .pt для дообучения, False - .yaml для обучения с нуля
        device: GPU id (0,1,..) или 'cpu', None - автоматический выбор
        weights_path: путь к зафайнтюненной модели для загрузки
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if weights_path:
            logger.info("Loading weights for YOLO from %r", weights_path)

            self.model = ultralytics.YOLO(weights_path)
        else:
            if finetune:
                model_file = "yolo" + f"{model_version}{model_size}.pt"
                logger.info("Loading trained %s on COCO dataset", model_file)
            else:
                model_file = f"{model_version}{model_size}.yaml"
                logger.info("Loading random initialized %s", model_file)

            self.model = ultralytics.YOLO(model_file)

    def train(self, config_path: str = None, save_dir: str = PATH_TO_SAVE_TRAIN_RUNS, **kwargs):
        """
        Обучение модели.
        - config_path: путь к YAML конфигу с гиперпараметрами
        - save_dir: папка для сохранения результатов (веса, графики, конфиг)
        - kwargs: любые параметры, которые перекрывают YAML конфиг
        """
        logger.info("Starting training on device: %r, save directory: %r", self.device, save_dir)

        config = {}
        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

        config.update(kwargs)

        if "name" not in config or config["name"] is None:
            config["name"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        if "device" not in config or config["device"] is None:
            config["device"] = self.device

        exp_path = Path(save_dir) / config["name"]
        exp_path.mkdir(parents=True, exist_ok=True)

        config_save_path = exp_path / "config_used.yaml"
        with open(config_save_path, "w") as f:
            yaml.dump(config, f)

        logger.debug("Full training config: %s", config)
        logger.info("Config saved to %r", config_save_path)

        try:
            self.model.train(**config)
        except Exception:
            logger.exception("YOLO train failed", exc_info=True)
            return None

        logger.info("Training finished. Weights saved in %s", exp_path / "weights")

    def predict(self, source, conf=0.01, imgsz=640, iou=0.25, save=False, save_dir="runs/predict"):
        """
        source: путь к изображению/папке/видео/стриму
        conf: confidence threshold
        imgsz: размер изображения
        save: сохранять ли результаты
        save_dir: папка для сохранения
        """
        try:
            results = self.model.predict(source=source, conf=conf, imgsz=imgsz, iou=iou)
        except Exception:
            logger.exception(f"Failed to process data from {source}", exc_info=True)
            return None

        if save:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            for i, res in enumerate(results):
                res.save(save_dir=save_path)
            logger.info(f"Detection results saved to {save_path}")
        return results

if __name__ == "__main__":
    setup_logging()
    create_default_config()
    