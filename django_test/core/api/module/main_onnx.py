from ultralytics import YOLO
import onnxruntime as ort
import cv2
import numpy as np
import json
import datetime
import locale
import os


#try:
#    locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')
#except:
#    locale.setlocale(locale.LC_TIME, 'Russian_Russia.1251')
#finally:
#    pass


class Pipeline:
    def __init__(self, yolo_model: str, classifier_model: str):
        self.detector = YOLO(yolo_model, task="detect")
        self.classifier = ort.InferenceSession(classifier_model, providers=['CPUExecutionProvider'])
        self.output_names = [output.name for output in self.classifier.get_outputs()]

    def process(self, image_path: str, classifier, output_json: str = "results.json", conf: float = 0.3, iou: float = 0.45, cropped_image_path: str = ""):
        results = self.detector(image_path, conf=conf, iou=iou, device="cpu")
        
        output = []
        image = cv2.imread(image_path)
        
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2]
            if crop is not None:
                photo_name = f"img_{datetime.datetime.now()}.png"
                cv2.imwrite(os.path.join(cropped_image_path, photo_name), crop)
            
            if crop.size == 0:
                continue

            crop_processed = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (320, 320))
            crop_processed = crop_processed.astype(np.float32) / 255.0
            
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            crop_processed = (crop_processed - mean) / std
            
            crop_processed = crop_processed.transpose(2, 0, 1)
            crop_processed = np.expand_dims(crop_processed, axis=0)
            
            outputs = self.classifier.run(None, {'input': crop_processed})
            classification_result = {}

            types = {'tree_type': {0: 'Береза', 1: 'Вяз', 2: 'Дуб', 3: 'Ель',
                        4: 'Ива', 5: 'Каштан', 6: 'Клен остролистный', 7: 'Клен ясенелистный',
                        8: 'Липа', 9: 'Лиственница', 10: 'Осина', 11: 'Рябина', 12: 'Сосна',
                        13: 'Туя', 14: 'Ясень', 15: 'неопределено'},
                        'has_hollow': {0: 'Нет', 1: 'Да'},
                        'has_cracks': {0: 'Нет', 1: 'Да'},
                        'has_fruits_or_flowers': {0: 'Нет', 1: 'Да'},
                        'has_rot': {0: 'Нет', 1: 'Да'},
                        'has_trunk_damage': {0: 'Нет', 1: 'Да'},
                        'has_crown_damage':{0: 'Нет', 1: 'Да'},
                        'overall_condition': {0: '',  1: 'Аварийное', 2: 'Нездоровое', 3: 'Нормальное', 4: 'Опасное', 5: 'Хорошее'}}
            
            for j, output_name in enumerate(self.output_names):
                pred = outputs[j][0]
                class_id = int(np.argmax(pred))
                confidence = float(np.max(pred))
                if output_name in types:
                    class_id = types[output_name][class_id]
                classification_result[output_name] = {
                    'class_id': class_id,
                    'confidence': confidence,
                    'probabilities': pred.tolist()
                }
            
            output.append({
                'bbox': [x1, y1, x2, y2],
                'detection_confidence': float(box.conf),
                'classification': classification_result,
                "photo_name": photo_name
            })
        
        answer = []
        for el in output:
            if "tree_type" not in classifier:
                d = datetime.datetime.now()
                answer.append({
                    "id": 0,
                    "plantName": d.strftime("%d %B %Y года, %H:%M"),
                    "probability": el["detection_confidence"] * 100,
                    "species": " ",
                    "trunkRot": el["classification"]["has_rot"]["class_id"],
                    "trunkHoles": el["classification"]["has_hollow"]["class_id"],
                    "trunkCracks": el["classification"]["has_cracks"]["class_id"],
                    "trunkDamage": el["classification"]["has_trunk_damage"]["class_id"],
                    "crownDamage": el["classification"]["has_crown_damage"]["class_id"],
                    "fruitingBodies": el["classification"]["has_fruits_or_flowers"]["class_id"],
                    "additionalInfo": "never",
                    "overallCondition": el["classification"]["overall_condition"]["class_id"],
                    "imageUrl": el["photo_name"],
                    "imagePath": "no",
                    "analyzedAt": d,
                    "isVerified": True
                })
            else:
                answer.append(el["classification"]["tree_type"]["class_id"])
        
        return answer

def run(image: str, output_json: str = "results.json", yolo: str = "yolov11m.pt", classifier: str = "resnet_classifier.onnx", conf: float = 0.3, iou: float = 0.45, cropped_image_path: str = ""):
    pipeline = Pipeline(yolo, classifier)
    results = pipeline.process(image, classifier, output_json=output_json, conf=conf, iou=iou, cropped_image_path=cropped_image_path)
    # print(results)
    return results

# if __name__ == "__main__":
#     run(
#         image=r"/home/e/Downloads/buk-derevo.jpg",
#         output_json="results.json",
#         yolo="/home/e/MegaDendrologAI/django_test/core/api/module/best.onnx",
#         classifier="/home/e/MegaDendrologAI/django_test/core/api/module/hollow_classification_small.onnx"
#     )
