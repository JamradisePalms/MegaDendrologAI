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
        
        for _, res in enumerate(results):
            res.save(save_dir=save_path)
            
        logger.info(f"Detection results saved to {save_path}")

    def save_cropped_images(results, image_path, save_dir, padding=10, class_names=None):
        """
        Вырезает области изображения по bounding box'ам и сохраняет их с отступом.

        Args:
            results: результаты YOLO-like
            image_path: путь к исходному изображению
            save_dir: директория для сохранения вырезанных изображений
            padding: отступ в пикселях вокруг bounding box'а
        """
        img = cv2.imread(str(image_path))
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if class_names is None:
            class_names = {
                0: "tree",
                1: "shrub"
            }
        
        for i, res in enumerate(results):
            boxes = res.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            cls_tensor = res.boxes.cls.cpu().numpy()
            
            for j, (box, cls) in enumerate(zip(boxes, cls_tensor)):
                x1, y1, x2, y2 = map(int, box)
                h, w = img.shape[:2]

                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)

                cropped = img[y1:y2, x1:x2]

                class_name = class_names[cls]

                output_path = save_path / f"{class_name}_{i}_obj_{j}.jpg"
                cv2.imwrite(str(output_path), cropped)