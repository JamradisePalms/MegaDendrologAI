from abc import ABC, abstractmethod
import cv2
import datetime
from configs.paths import PathConfig
import logging
import yaml 
from pathlib import Path

PATH_TO_SAVE_TRAIN_RUNS = PathConfig.ML.Detection.PATH_TO_SAVE_TRAIN_RUNS
logger = logging.getLogger(__name__)

class BaseDetection(ABC):
    def __init__(self, weights_path: str = None, config_path: str = None, device: str = None):
        self.weights_path = weights_path
        self.device = device

        self.config = {
            "conf": 0.3,
            "imgsz": 640,
            "iou": 0.4,
            "batch": 8
        }

        if config_path:
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}
            self.config.update(yaml_config)

    @property
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def predict(self, image_path, **kwargs):
        pass
        
    def train(self, config_path: str = None, save_dir: str = PATH_TO_SAVE_TRAIN_RUNS, **kwargs):
        """
        Обучение модели.
        - config_path: путь к YAML конфигу с гиперпараметрами обучения
        - save_dir: папка для сохранения результатов (веса, графики, конфиг)
        - kwargs: любые параметры, которые перекрывают YAML конфиг
        """
        logger.info("Starting training on device: %r, save directory: %r", self.device, save_dir)

        config = {**self.config}
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
    
    def save_results(self, results, save_dir):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if isinstance(results, list):
            for i, res in enumerate(results):
                res.save(filename=save_path / f"detection_{i}.jpg")
                
                if hasattr(res, 'save_txt'):
                    res.save_txt(save_path / f"detection_{i}.txt")
                    
        else:
            results.save(filename=save_path / "detection_0.jpg")
            if hasattr(results, 'save_txt'):
                results.save_txt(save_path / "detection_0.txt")
        
        logger.info(f"Detection results saved to {save_path}")

    def save_cropped_images(self, results, original_image_path, save_dir):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        original_img = cv2.imread(str(original_image_path))
        
        if isinstance(results, list):
            result = results[0]
        else:
            result = results
            
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = map(int, box.tolist())
                
                cropped_img = original_img[y1:y2, x1:x2]
                
                cv2.imwrite(str(save_path / f"cropped_{i}.jpg"), cropped_img)
        
        logger.info(f"Cropped images saved to {save_path}")
