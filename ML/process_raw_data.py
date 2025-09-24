import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image
import shutil
from tqdm import tqdm

from ML.Detection.YOLOWrapper import YoloWrapper
from ML.Classification.vision_classifier import GptClassifier
from configs.paths import PathConfig

PATH_TO_SAVE_PROCESSED_IMAGES  = PathConfig.ML.PATH_TO_SAVE_PROCESSED_IMAGES

class DatasetLabeler:
    def __init__(self, yolo_weights_path: str, output_dir: Path = None):
        """
        Инициализация пайплайна разметки
        
        Args:
            yolo_weights_path: путь к обученным весам YOLO
            output_dir: директория для сохранения результатов
        """
        self.yolo = YoloWrapper(weights_path=yolo_weights_path)
        self.classifier = GptClassifier()
        
        self.output_dir = output_dir or PathConfig.ML.PATH_TO_SAVE_PROCESSED_IMAGES
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.crops_dir = self.output_dir / "tree_crops"
        self.visualization_dir = self.output_dir / "visualization"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.crops_dir, self.visualization_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def detect_and_crop_trees(self, 
                            image_paths: List[Path], 
                            confidence_threshold: float = 0.3,
                            save_visualization: bool = True,
                            imgsz: int = 640) -> Dict[str, Any]:
        """
        Детекция деревьев и вырезание регионов
        
        Args:
            image_paths: список путей к изображениям
            confidence_threshold: порог уверенности для детекции
            save_visualization: сохранять ли изображения с визуализацией
        
        Returns:
            Словарь с метаданными детекций
        """
        detection_metadata = {
            "total_images": len(image_paths),
            "total_trees_detected": 0,
            "detections_per_image": {},
            "crop_info": []
        }
        
        crop_counter = 0
        
        for image_path in tqdm(image_paths, desc="Detecting trees"):
            if not image_path.exists():
                print(f"Warning: {image_path} не существует")
                continue
            
            try:
                results = self.yolo.predict(
                    source=str(image_path),
                    conf=confidence_threshold,
                    save=False,
                    verbose=False,
                    imgsz=imgsz,
                )
            except Exception as e:
                print(f"Skiped image {image_path} because of {e}")
                continue
            
            if not results:
                detection_metadata["detections_per_image"][image_path.name] = 0
                continue
            
            for result_idx, result in enumerate(results):
                image = cv2.imread(str(image_path))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    detection_metadata["detections_per_image"][image_path.name] = 0
                    continue
                
                detection_metadata["detections_per_image"][image_path.name] = len(boxes)
                detection_metadata["total_trees_detected"] += len(boxes)
                
                if save_visualization:
                    vis_image = self._draw_detections(image_rgb.copy(), boxes, result.names)
                    vis_path = self.visualization_dir / f"det_{image_path.stem}.jpg"
                    cv2.imwrite(str(vis_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                
                for box_idx, box in enumerate(boxes):
                    crop, bbox = self._crop_tree_region(image_rgb, box.xyxy[0].cpu().numpy())
                    
                    if crop is not None:
                        crop_filename = f"tree_{crop_counter:06d}_{image_path.stem}.jpg"
                        crop_path = self.crops_dir / crop_filename
                        
                        cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                        
                        detection_metadata["crop_info"].append({
                            "crop_id": crop_counter,
                            "source_image": image_path.name,
                            "crop_filename": crop_filename,
                            "bbox": bbox.tolist(),
                            "confidence": float(box.conf[0].cpu().numpy()),
                            "class_id": int(box.cls[0].cpu().numpy()),
                            "class_name": result.names[int(box.cls[0].cpu().numpy())]
                        })
                        
                        crop_counter += 1
        
        metadata_path = self.metadata_dir / "detection_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(detection_metadata, f, indent=4, ensure_ascii=False)
        
        print(f"Детекция завершена. Обнаружено {detection_metadata['total_trees_detected']} деревьев")
        return detection_metadata

    def classify_tree_crops(self, crop_metadata: List[Dict] = None) -> Dict[str, Any]:
        """
        Классификация вырезанных деревьев с помощью GPT
        
        Args:
            crop_metadata: метаданные вырезанных регионов (если None, загружаем из файла)
        
        Returns:
            Словарь с результатами классификации
        """
        if crop_metadata is None:
            metadata_path = self.metadata_dir / "detection_metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                detection_metadata = json.load(f)
                crop_metadata = detection_metadata["crop_info"]
        
        classification_results = {
            "total_crops_classified": len(crop_metadata),
            "classification_results": []
        }
        
        crop_paths = [self.crops_dir / item["crop_filename"] for item in crop_metadata]
        
        gpt_results = self.classifier.run(crop_paths)
        
        for i, (crop_info, gpt_result) in enumerate(zip(crop_metadata, gpt_results)):
            classification_results["classification_results"].append({
                **crop_info,
                "gpt_classification": gpt_result
            })
        
        classification_path = self.metadata_dir / "classification_results.json"
        with open(classification_path, 'w', encoding='utf-8') as f:
            json.dump(classification_results, f, indent=4, ensure_ascii=False)
        
        print(f"Классификация завершена. Обработано {len(crop_metadata)} изображений")
        return classification_results

    def _crop_tree_region(self, image: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вырезание региона дерева из изображения
        
        Args:
            image: исходное изображение
            bbox: bounding box [x1, y1, x2, y2]
        
        Returns:
            Вырезанное изображение и нормализованный bbox
        """
        h, w = image.shape[:2]
        
        if bbox.max() <= 1.0:
            bbox = bbox * np.array([w, h, w, h])
        
        x1, y1, x2, y2 = bbox.astype(int)
        
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None, bbox
        
        return crop, np.array([x1, y1, x2, y2])

    def _draw_detections(self, image: np.ndarray, boxes, class_names: Dict) -> np.ndarray:
        """
        Рисование bounding boxes на изображении
        
        Args:
            image: исходное изображение
            boxes: детекции от YOLO
            class_names: словарь с именами классов
        
        Returns:
            Изображение с нарисованными детекциями
        """
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_names[class_id]}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return image

    def run_full_pipeline(self, 
                         image_directory: Path,
                         image_extensions: List[str] = None) -> Dict[str, Any]:
        """
        Полный пайплайн разметки
        
        Args:
            image_directory: директория с изображениями
            image_extensions: расширения изображений (по умолчанию ['.jpg', '.jpeg', '.png'])
        
        Returns:
            Объединенные результаты разметки
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(image_directory.glob(f"*{ext}")))
            image_paths.extend(list(image_directory.glob(f"**/*{ext}")))
        
        print(f"Найдено {len(image_paths)} изображений")
        
        if not image_paths:
            raise ValueError(f"В директории {image_directory} не найдено изображений")
        
        print("=== Шаг 1: Детекция деревьев ===")
        detection_metadata = self.detect_and_crop_trees(image_paths)
        
        print("\n=== Шаг 2: Классификация деревьев ===")
        classification_results = self.classify_tree_crops()
        
        final_results = {
            "pipeline_info": {
                "input_directory": str(image_directory),
                "output_directory": str(self.output_dir),
                "total_images_processed": len(image_paths)
            },
            "detection_results": detection_metadata,
            "classification_results": classification_results
        }
        
        final_path = self.metadata_dir / "final_labeling_results.json"
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== Пайплайн завершен ===")
        print(f"Результаты сохранены в: {self.output_dir}")
        print(f"Деревья обнаружены: {detection_metadata['total_trees_detected']}")
        print(f"Деревья классифицированы: {classification_results['total_crops_classified']}")
        
        return final_results


if __name__ == "__main__":
    # TODO: need debugging 
    
    PATH_TO_YOLO_WEIGHTS = "/home/jamradise/MegaDendrologAI/ML/Detection/yolo_train/runs/train/20250923_195225/weights/best.pt"
    IMAGE_DIRECTORY = PathConfig.ML.PATH_TO_RAW_IMAGES
    
    labeler = DatasetLabeler(yolo_weights_path=str(PATH_TO_YOLO_WEIGHTS))
    
    # results = labeler.run_full_pipeline(image_directory=IMAGE_DIRECTORY)
    
    # 1. Только детекция
    detection_results = labeler.detect_and_crop_trees(list(IMAGE_DIRECTORY.glob("*.jpg")))
    
    # 2. Только классификация
    # classification_results = labeler.classify_tree_crops()