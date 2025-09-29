from ultralytics import YOLO
import onnxruntime as ort
import cv2
import numpy as np
import json

class Pipeline:
    def __init__(self, yolo_model: str, classifier_model: str):
        self.detector = YOLO(yolo_model, task="detect")
        self.classifier = ort.InferenceSession(classifier_model, providers=['CPUExecutionProvider'])
        self.output_names = [output.name for output in self.classifier.get_outputs()]

    def process(self, image_path: str, output_json: str = "results.json", conf: float = 0.3, iou: float = 0.45):
        results = self.detector(image_path, conf=conf, iou=iou, device="cpu")
        
        output = []
        image = cv2.imread(image_path)
        
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            crop_processed = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (224, 224))
            crop_processed = crop_processed.astype(np.float32) / 255.0
            
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            crop_processed = (crop_processed - mean) / std
            
            crop_processed = crop_processed.transpose(2, 0, 1)
            crop_processed = np.expand_dims(crop_processed, axis=0)
            
            outputs = self.classifier.run(None, {'input': crop_processed})
            classification_result = {}
            
            for j, output_name in enumerate(self.output_names):
                pred = outputs[j][0]
                class_id = int(np.argmax(pred))
                confidence = float(np.max(pred))
                
                classification_result[output_name] = {
                    'class_id': class_id,
                    'confidence': confidence,
                    'probabilities': pred.tolist()
                }
            
            output.append({
                'bbox': [x1, y1, x2, y2],
                'detection_confidence': float(box.conf),
                'classification': classification_result
            })
        
        with open(output_json, "w") as f:
            json.dump(output, f, indent=2)
        
        return output

def run(image: str, output_json: str = "results.json", yolo: str = "yolov11m.pt", classifier: str = "resnet_classifier.onnx", conf: float = 0.3, iou: float = 0.45):
    pipeline = Pipeline(yolo, classifier)
    results = pipeline.process(image, output_json=output_json, conf=conf, iou=iou)
    print(results)
    return results

if __name__ == "__main__":
    run(
        image=r"C:\Users\shari\PycharmProjects\MegaDendrologAI\dataset_to_mark\82x7vXkkkVQ.jpg",
        output_json="results.json",
        yolo=r"C:\Users\shari\Downloads\yolo11m_on_new_data\best.onnx",
        classifier=r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Classification\results\saved_models\hollow_classification_small.onnx"
    )