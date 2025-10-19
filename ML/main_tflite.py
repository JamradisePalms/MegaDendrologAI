import tensorflow as tf
import cv2
import numpy as np
import json
from ultralytics import YOLO

def softmax(x: np.ndarray) -> np.ndarray:
    if x.ndim != 1:
        x = x.reshape(-1)
    return np.exp(x) / np.sum(np.exp(x))

mapping_dict = {
    'tree_type': {0: 'He определено', 1: 'Береза', 2: 'Боярышник', 3: 'Вяз', 4: 'Дерен белый', 5: 'Дуб', 6: 'Ель', 7: 'Ива', 8: 'Карагана древовидная', 9: 'Кизильник', 10: 'Клен остролистный', 11: 'Клен ясенелистный', 12: 'Лапчатка кустарниковая', 13: 'Лещина', 14: 'Липа', 15: 'Лиственница', 16: 'Осина', 17: 'Пузыреплодник калинолистный', 18: 'Роза морщинистая', 19: 'Роза собачья', 20: 'Рябина', 21: 'Сирень обыкновенная', 22: 'Сосна', 23: 'Спирея', 24: 'Туя', 25: 'Чубушник', 26: 'Ясень'},
    'has_hollow': {0: 'No', 1: 'Yes'},            
    'has_cracks': {0: 'No', 1: 'Yes'},            
    'has_fruits_or_flowers': {0: 'No', 1: 'Yes'},            
    'has_rot': {0: 'No', 1: 'Yes'},            
    'has_trunk_damage': {0: 'No', 1: 'Yes'},            
    'has_crown_damage':{0: 'No', 1: 'Yes'},
    'dry_branch_percentage': {0: 'Normal', 1: 'Dry', 2: 'Very Dry', 3: 'Extremely Dry'},
    'overall_condition': {0: '',  1: 'Аварийное', 2: 'Нездоровое', 3: 'Нормальное', 4: 'Опасное', 5: 'Хорошее'}
}

class PipelineTFLite:
    def __init__(self, yolo_model: str, classifier_model: str):
        self.detector = YOLO(yolo_model, task="detect")
        
        # Загрузка TFLite модели
        self.interpreter = tf.lite.Interpreter(model_path=classifier_model)
        self.interpreter.allocate_tensors()
        
        # Получение информации о входе и выходе
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"TFLite input shape: {self.input_details[0]['shape']}")
        print(f"TFLite input dtype: {self.input_details[0]['dtype']}")
        print(f"TFLite output details: {[out['name'] for out in self.output_details]}")
        
        # Предполагаем, что выходы имеют те же имена, что и в ONNX версии
        # Если имена отличаются, возможно потребуется mapping
        self.output_names = [out['name'] for out in self.output_details]

    def process(self, image_path: str, output_json: str = "results.json", conf: float = 0.3, iou: float = 0.45, resize=320):
        results = self.detector(image_path, conf=conf, iou=iou, device="cpu")
        
        output = []
        image = cv2.imread(image_path)
        
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Preprocessing для TFLite модели
            crop_processed = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (resize, resize))
            crop_processed = crop_processed.astype(np.float32) / 255.0
            
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            crop_processed = (crop_processed - mean) / std
            
            # TFLite обычно ожидает NHWC формат, но проверим входные данные
            if len(self.input_details[0]['shape']) == 4:
                if self.input_details[0]['shape'][1] == 3:  # NCHW формат
                    crop_processed = crop_processed.transpose(2, 0, 1)
            
            crop_processed = np.expand_dims(crop_processed, axis=0)
            
            # Инференс через TFLite
            self.interpreter.set_tensor(self.input_details[0]['index'], crop_processed)
            self.interpreter.invoke()
            
            # Получение результатов
            classification_result = {}
            for j, output_detail in enumerate(self.output_details):
                output_data = self.interpreter.get_tensor(output_detail['index'])
                pred = output_data[0]  # Берем первый элемент батча
                
                # Определяем имя выхода (используем mapping_dict ключи если подходят)
                output_name = self.output_names[j] if j < len(self.output_names) else f"output_{j}"
                
                # Если имя выхода есть в mapping_dict, используем его, иначе используем имя из модели
                if output_name in mapping_dict:
                    mapping_key = output_name
                else:
                    # Попробуем найти подходящий ключ в mapping_dict
                    mapping_key = next((key for key in mapping_dict.keys() if key in output_name), output_name)
                
                class_id = int(np.argmax(pred))
                confidence = float(np.max(pred))
                
                # Используем mapping_dict если ключ существует, иначе используем числовые ID
                if mapping_key in mapping_dict:
                    class_name = mapping_dict[mapping_key].get(class_id, f"Unknown_{class_id}")
                else:
                    class_name = f"Class_{class_id}"
                
                classification_result[output_name] = {
                    'class_id': class_name,
                    'confidence': confidence,
                    'probabilities': (softmax(pred) * 100).tolist()
                }
            
            print(classification_result, end='\n\n\n')
            output.append({
                'bbox': [x1, y1, x2, y2],
                'detection_confidence': float(box.conf),
                'classification': classification_result
            })
            
        # with open(output_json, "w") as f:
        #     json.dump(output, f, indent=2)
        
        return output

def run_tflite(image: str, output_json: str = "results.json", yolo: str = "yolov11m.pt", 
               classifier: str = "model.tflite", conf: float = 0.2, iou: float = 0.45, resize=320):
    pipeline = PipelineTFLite(yolo, classifier)
    results = pipeline.process(image, output_json=output_json, conf=conf, iou=iou, resize=resize)
    print(results)
    return results

if __name__ == "__main__":
    run_tflite(
        image=r"C:\Users\shari\OneDrive\Рабочий стол\photo_2025-10-18_20-38-08.jpg",
        output_json="results_tflite.json",
        yolo=r"C:\Users\shari\Downloads\yolo11m_on_new_data\best.onnx",
        classifier=r"C:\Users\shari\Downloads\Telegram Desktop\model.tflite",
        resize=224
    )