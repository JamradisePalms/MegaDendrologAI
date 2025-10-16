from ultralytics import YOLO
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import onnxruntime as ort
import cv2
import numpy as np
import json
import os

class State(TypedDict):
    is_cropped_by_user: bool
    image_path: str
    cropped_image: str
    vlm_validate: bool
    val_steps: int
    vlm_verdict: bool
    detection_json: dict
    classification_json: dict


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=np.float64)
    if x.ndim == 1:
        x = x - np.max(x)
        e = np.exp(x)
        return (e / e.sum()).astype(np.float32)
    else:
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


mapping_dict = {
    'tree_type': {0: 'He определено', 1: 'Береза', 2: 'Боярышник', 3: 'Вяз', 4: 'Дерен белый', 5: 'Дуб', 6: 'Ель', 7: 'Ива', 8: 'Карагана древовидная', 9: 'Кизильник', 10: 'Клен остролистный', 11: 'Клен ясенелистный', 12: 'Лапчатка кустарниковая', 13: 'Лещина', 14: 'Липа', 15: 'Лиственница', 16: 'Осина', 17: 'Пузыреплодник калинолистный', 18: 'Роза морщинистая', 19: 'Роза собачья', 20: 'Рябина', 21: 'Сирень обыкновенная', 22: 'Сосна', 23: 'Спирея', 24: 'Туя', 25: 'Чубушник', 26: 'Ясень'},
    'has_hollow': {0: 'No', 1: 'Yes'},
    'has_cracks': {0: 'No', 1: 'Yes'},
    'has_fruits_or_flowers': {0: 'No', 1: 'Yes'},
    'has_rot': {0: 'No', 1: 'Yes'},
    'has_trunk_damage': {0: 'No', 1: 'Yes'},
    'has_crown_damage': {0: 'No', 1: 'Yes'},
    'dry_branch_percentage': {0: 'Normal', 1: 'Dry', 2: 'Very Dry', 3: 'Extremely Dry'},
    'overall_condition': {0: '',  1: 'Аварийное', 2: 'Нездоровое', 3: 'Нормальное', 4: 'Опасное', 5: 'Хорошее'}
}


class Pipeline:
    def __init__(self, yolo_model: str, classifier_model: str, vlm_model=None, device="cpu"):
        self.detector = YOLO(yolo_model)
        self.classifier = ort.InferenceSession(classifier_model, providers=["CPUExecutionProvider"])
        self.vlm = vlm_model
        self.device = device

        self.output_names = [o.name for o in self.classifier.get_outputs()]
        self.input_name = self.classifier.get_inputs()[0].name

        self._run_params = {
            "conf": 0.3,
            "iou": 0.45,
            "resize": 320,
            "device": device,
            "max_vlm_attempts": 5
        }

        graph = StateGraph(State)
        graph.add_node("start", lambda state: state)
        graph.add_node("detection", self.detect)
        graph.add_node("classification", self.classify)
        graph.add_node("vlm_validation", self.validation_node)

        graph.add_edge(START, "start")
        graph.add_edge("start", "detection")
        graph.add_edge("detection", "classification")
        graph.add_edge("classification", "vlm_validation")
        graph.add_edge("vlm_validation", END)

        self.graph = graph.compile()

    def detect(self, state: State) -> State:
        if state.get("is_cropped_by_user", False):
            image = cv2.imread(state["image_path"])
            h, w = image.shape[:2]
            state["detection_json"] = {"detections": [{"bbox": [0, 0, w, h], "confidence": 1.0}]}
            return state

        conf = self._run_params["conf"]
        iou = self._run_params["iou"]
        device = self._run_params["device"]

        results = self.detector(state["image_path"], conf=conf, iou=iou, device=device)
        detection_list = []
        for box in results[0].boxes:
            xy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(map(float, box.xyxy[0]))
            x1, y1, x2, y2 = map(int, xy)
            conf_val = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf)
            detection_list.append({"bbox": [x1, y1, x2, y2], "confidence": conf_val})

        state["detection_json"] = {"detections": detection_list}
        return state

    def classify(self, state: State) -> State:
        img = cv2.imread(state["image_path"])
        if img is None:
            raise FileNotFoundError(f"Image not found: {state['image_path']}")

        resize = self._run_params.get("resize", 320)
        detections = state.get("detection_json", {}).get("detections", [])
        if not detections:
            h, w = img.shape[:2]
            detections = [{"bbox": [0, 0, w, h], "confidence": 1.0}]
            state["detection_json"] = {"detections": detections}

        results_out = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (resize, resize), interpolation=cv2.INTER_LINEAR)
            crop_processed = crop_resized.astype(np.float32) / 255.0

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            crop_processed = (crop_processed - mean[None, None, :]) / std[None, None, :]
            crop_processed = np.transpose(crop_processed, (2, 0, 1))
            crop_processed = np.expand_dims(crop_processed, axis=0)
            crop_processed = np.ascontiguousarray(crop_processed.astype(np.float32))

            try:
                outputs = self.classifier.run(None, {self.input_name: crop_processed})
            except Exception as e:
                raise RuntimeError(f"ONNX inference failed: {e}")

            classification_result = {}
            for j, output_name in enumerate(self.output_names):
                pred = outputs[j]
                pred_vec = pred[0] if pred.ndim == 2 else pred
                class_id = int(np.argmax(pred_vec))
                confidence = float(np.max(pred_vec))
                probs = softmax(pred_vec)
                classification_result[output_name] = {
                    "class_label": mapping_dict.get(output_name, {}).get(class_id, str(class_id)),
                    "confidence": confidence,
                    "probabilities": (probs * 100).tolist()
                }

            results_out.append({
                "bbox": det["bbox"],
                "classification": classification_result
            })

        state["classification_json"] = {"results": results_out}
        return state

    def validation_node(self, state: State) -> State:
        """
        VLM должна уметь менять state (валидировать его и фиксить баги в репорте).
        Тут в цикле бегает до хорошего вердикта или до max_attempts.
        """
        max_attempts = self._run_params.get("max_vlm_attempts", 5)
        state.setdefault("val_steps", 0)
        if not state.get("vlm_validate", False):
            state["vlm_verdict"] = True
            return state

        attempt = state.get("val_steps", 0)
        while attempt < max_attempts:
            if self.vlm is not None:
                verdict = self.vlm.check_classification_quality(state)
            else:
                verdict = "good" if np.random.rand() > 0.2 else "bad"

            if isinstance(verdict, str):
                is_good = verdict.lower() in ("good", "ok", "true", "1")
            else:
                is_good = bool(verdict)

            if is_good:
                state["vlm_verdict"] = True
                state["val_steps"] = attempt
                return state
            else:
                attempt += 1
                state["val_steps"] = attempt
                state["vlm_verdict"] = False
                state = self.classify(state)

        state["vlm_verdict"] = False
        return state

    def process(self, image_path: str, output_json: str = "results.json", conf: float = 0.3, iou: float = 0.45, resize=320, device="cpu", vlm_validate=False, max_vlm_attempts: int = 5, is_cropped_by_user=False):
        self._run_params.update({
            "conf": conf,
            "iou": iou,
            "resize": resize,
            "device": device,
            "max_vlm_attempts": max_vlm_attempts
        })

        init_state: State = {
            "is_cropped_by_user": is_cropped_by_user,
            "image_path": image_path,
            "cropped_image": "",
            "vlm_validate": vlm_validate,
            "val_steps": 0,
            "vlm_verdict": False,
            "detection_json": {},
            "classification_json": {}
        }

        final_state = self.graph.invoke(init_state)

        if output_json:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(final_state.get("classification_json", {}), f, indent=2, ensure_ascii=False)

        return final_state


def run(
    image: str,
    output_json: str = "results.json",
    yolo: str = "yolov11m.pt",
    classifier: str = "resnet_classifier.onnx",
    is_cropped_by_user: bool = False,
    conf: float = 0.3,
    iou: float = 0.45,
    resize=320,
    vlm=None,
    device="cpu",
    vlm_validate=False,
    max_vlm_attempts=5
):
    pipeline = Pipeline(yolo, classifier, vlm_model=vlm, device=device)
    results = pipeline.process(image_path=image, output_json=output_json, conf=conf, iou=iou, resize=resize, device=device, vlm_validate=vlm_validate, max_vlm_attempts=max_vlm_attempts, is_cropped_by_user=is_cropped_by_user)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results


if __name__ == "__main__":
    run(
        image=r"C:\Users\shari\OneDrive\Рабочий стол\photo_2025-10-16_19-36-34.jpg",
        output_json="results.json",
        yolo=r"C:\Users\shari\Downloads\yolo11m_on_new_data\best.onnx",
        classifier=r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Classification\results\saved_models\best_all_classes_53\simple_model.onnx",
        resize=320,
        conf=0.2,
        iou=0.45,
        device="cpu",
        vlm=None,
        vlm_validate=True,
        max_vlm_attempts=3
    )
