import ultralytics
import yaml
from pathlib import Path
import torch
from datetime import datetime

class YOLOWrapper:
    def __init__(self, model_version: str = 'v11', model_size: str = 'n', finetune: bool = True, device=None, weights_path: str = None):
        """
        model_version: версия YOLO (например, 'v11')
        model_size: размер модели ('n', 's', 'm', 'l', 'x')
        finetune: True - использовать .pt для дообучения, False - .yaml для обучения с нуля
        device: GPU id (0,1,..) или 'cpu', None - автоматический выбор
        weights_path: путь к зафайнтюненной модели для загрузки
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if weights_path:
            self.model = ultralytics.YOLO(weights_path)
        else:
            model_file = f"{model_version}{model_size}.pt" if finetune else f"{model_version}{model_size}.yaml"
            self.model = ultralytics.YOLO(model_file)

    def train(self, config_path: str = None, save_dir: str = "runs/train", **kwargs):
        """
        Обучение модели.
        - config_path: путь к YAML конфигу с гиперпараметрами
        - save_dir: папка для сохранения результатов (веса, графики, конфиг)
        - kwargs: любые параметры, которые перекрывают YAML конфиг
        """
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

        self.model.train(**config)

        print(f"\n Training finished. Weights saved in {exp_path / 'weights'}")

    def predict(self, source, conf=0.25, imgsz=640, save=False, save_dir="runs/predict", verbose=True):
        """
        source: путь к изображению/папке/видео/стриму
        conf: confidence threshold
        imgsz: размер изображения
        save: сохранять ли результаты
        save_dir: папка для сохранения
        """
        results = self.model.predict(source=source, conf=conf, imgsz=imgsz)
        if save:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            for i, res in enumerate(results):
                res.save(save_dir=save_path)
        if verbose:
            for res in results:
                res.print()
        return results

    @staticmethod
    def create_default_config(path="yolo_config.yaml"):
        """
        Создаёт YAML конфиг с базовыми параметрами обучения.
        """
        default_config = {
            "data": "data.yaml",
            "epochs": 100,
            "imgsz": 640,
            "batch": 16,
            "device": None,
            "workers": 8,
            "optimizer": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "patience": 50,
            "resume": False,
            "freeze": 0,
            "pretrained": True,
            "save": True,
            "save_period": -1,
            "verbose": True,
            "plots": True,
            "project": "runs/train",
            "name": None,
            "exist_ok": False,
            "cache": False,
            "rect": False,
            "cos_lr": False,
            "close_mosaic": 10,
            "amp": True
        }
        with open(path, "w") as f:
            yaml.dump(default_config, f)
        print(f"Default config saved to {path}")

if __name__ == "__main__":
    YOLOWrapper.create_default_config("my_yolo_config.yaml")

    wrapper = YOLOWrapper(model_version="v11", model_size="n", finetune=True)
    # wrapper.train(config_path="my_yolo_config.yaml", epochs=50, batch=32)

    # finetuned = YOLOWrapper(weights_path="runs/train/20250921_123456/weights/best.pt")

    results = wrapper.predict(r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Detection\data", conf=0.3, save=True)