from pathlib import Path
from typing import Iterable, List
import shutil
import os
import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from ML.Classification.vlm_lib import BaseClassifier
from ML.Classification.vlm_lib.Dataclasses import ClassificationTreeAnalysis
from ML.Detection.BaseDetection import BaseDetection
from ML.Classification.vlm_lib.utils import write_json
from ML.Classification.vlm_lib.QwenImageClassifier import QwenImageClassifier
from ML.Detection.YOLOWrapper import YoloWrapper
from configs.paths import PathConfig

CLASSIFICATION_PATHS = PathConfig.ML.Classification
CLASSIFICATION_PROMPT_FILEPATH = CLASSIFICATION_PATHS.CLASSIFICATION_PROMPT_FILEPATH

YOLO_MODEL = PathConfig.ML.Detection.PATH_TO_YOLO_MEDIUM_BEST_WEIGHTS
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(
        self,
        detection_model: BaseDetection,
        classifier: BaseClassifier,
        path_to_save_final_json: Path | str = None,
        max_workers: int = 1,
        clean_up_tmp_files = True,
    ):
        self.detectron = detection_model
        self.classifier = classifier
        self.path_to_save_final_json = path_to_save_final_json
        self.max_workers = max_workers
        self.cleanup_temp_files = clean_up_tmp_files

    def detect_objects(self, image_path: Path | str = None, **kwargs):
        cur_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tmp_folder_name = Path(f"tmp_{cur_date}")

        try:
            results = self.detectron.predict(image_path, **kwargs)
            self.detectron.save_results(results, tmp_folder_name / "detection_results")
            self.detectron.save_cropped_images(results, image_path, 
                                             save_dir=tmp_folder_name / "cropped")
            return results, tmp_folder_name
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {e}")
            if tmp_folder_name.exists():
                shutil.rmtree(tmp_folder_name)
            raise
    
    def classify_objects(self, image_path: Path | str = None):
        try:
            return self.classifier.run(image_path)
        except Exception as e:
            logger.error(f"Classification failed for {image_path}: {e}")
            return None
    
    def _process_one_image(self, image_path, **kwargs):
        """Обработка одного изображения: детекция + классификация"""
        try:
            _, tmp_folder_name = self.detect_objects(
                image_path=image_path, **kwargs
            )
            
            classification_results = []
            cropped_dir = tmp_folder_name / "cropped"
            
            if cropped_dir.exists():
                for cropped_image in cropped_dir.iterdir():
                    if cropped_image.is_file() and cropped_image.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        result = self.classify_objects(image_path=cropped_image)
                        if result:
                            classification_results.append({
                                'original_image': str(image_path),
                                'cropped_image': str(cropped_image),
                                'classification_result': result
                            })
            
            if self.cleanup_temp_files:
                self._cleanup_tmp_folder(tmp_folder_name)
            
            return classification_results
            
        except Exception as e:
            logger.error(f"Processing failed for {image_path}: {e}")
            return []

    def run(self, image_path: Iterable[Path | str] | Path, **kwargs):
        """Основной метод запуска пайплайна"""
        run_results = []
        
        if not isinstance(image_path, (list, tuple)) and not hasattr(image_path, '__iter__'):
            images = [image_path]
        else:
            images = list(image_path)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_image = {
                executor.submit(self._process_one_image, image, **kwargs): image 
                for image in images
            }
            
            for future in as_completed(future_to_image):
                image = future_to_image[future]
                try:
                    result = future.result()
                    run_results.extend(result)
                except Exception as e:
                    logger.error(f"Error processing {image}: {e}")

        if self.path_to_save_final_json and run_results:
            write_json(run_results, self.path_to_save_final_json)
            logger.info(f"Saved {len(run_results)} results to {self.path_to_save_final_json}")
        
        return run_results

    def _cleanup_tmp_folder(self, tmp_folder: Path):
        """Рекурсивно удаляет временную папку"""
        try:
            import shutil
            if tmp_folder.exists():
                shutil.rmtree(tmp_folder)
                logger.info(f"Cleaned up temporary folder: {tmp_folder}")
        except Exception as e:
            logger.warning(f"Could not clean up {tmp_folder}: {e}")


if __name__ == "__main__":
    classifier = QwenImageClassifier(CLASSIFICATION_PROMPT_FILEPATH, ClassificationTreeAnalysis)
    detectron = YoloWrapper(weights_path=YOLO_MODEL)
    path_to_save_final_json = "./results.json"

    pipe = Pipeline(detection_model=detectron, classifier=classifier, path_to_save_final_json=path_to_save_final_json)

    pipe.run({
        'iou': 0.45,
        'conf': 0.3, 
        'imgsz': 640,
    })
