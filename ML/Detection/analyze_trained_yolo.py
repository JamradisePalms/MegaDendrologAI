import yaml
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from configs.paths import PathConfig

PATH_TO_DATA = PathConfig.ML.Detection.PATH_TO_SAVE_DATASET / "Tree-Quality-3-7/data.yaml"
WEIGHTS_PATH = "/home/jamradise/MegaDendrologAI/ML/Detection/yolo_train/runs/train/20250922_185054/weights/best.pt"

def find_best_conf_iou(model, data_path, conf_range=np.arange(0.05, 0.96, 0.05), iou_range=np.arange(0.3, 0.71, 0.1)):
    """Подбирает оптимальные conf и iou по максимальному F1-score"""
    best_f1 = -1
    best_conf = 0.25
    best_iou = 0.45
    results = []
    
    print(f"Поиск оптимальных параметров для {len(conf_range)*len(iou_range)} комбинаций...")
    for conf in conf_range:
        for iou in iou_range:
            print(f"Тестируем conf={conf:.2f}, iou={iou:.2f}...")
            metrics = model.val(
                data=data_path,
                conf=conf,
                iou=iou,
                verbose=False
            )
            
            if hasattr(metrics.box, 'f1') and metrics.box.f1 is not None:
                avg_f1 = metrics.box.f1.mean()
            else:
                precision = metrics.box.precision.mean()
                recall = metrics.box.recall.mean()
                avg_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append((conf, iou, avg_f1))
            print(f"    F1-score: {avg_f1:.4f}")
            
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_conf = conf
                best_iou = iou
    
    print("\nРезультаты поиска:")
    for conf, iou, f1 in sorted(results, key=lambda x: x[2], reverse=True)[:5]:
        print(f"  conf={conf:.2f}, iou={iou:.2f} → F1={f1:.4f}")
    
    print(f"\nЛучшие параметры: conf={best_conf:.2f}, iou={best_iou:.2f} (F1={best_f1:.4f})")
    return best_conf, best_iou

def analyze_box_sizes(model, data_path, conf, iou, imgsz=640):
    """Анализирует размеры предсказанных боксов при заданных параметрах"""
    val_dir = Path(data_path)
    image_paths = list(val_dir.glob("*.jpg"))

    sizes = []
    for img_path in image_paths:
        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False
        )
        for res in results:
            for box in res.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                sizes.append(width * height)
    
    if not sizes:
        print("Не найдено ни одного бокса — попробуйте уменьшить conf")
        return
    
    print("\nСТАТИСТИКА РАЗМЕРОВ БОКСОВ:")
    print(f"Всего боксов: {len(sizes)}")
    print(f"Средний размер: {np.mean(sizes):.1f} пикселей²")
    print(f"Минимальный размер: {np.min(sizes):.1f} пикселей²")
    print(f"Максимальный размер: {np.max(sizes):.1f} пикселей²")
    print(f"Медианный размер: {np.median(sizes):.1f} пикселей²")
    print(f"95-й перцентиль: {np.percentile(sizes, 95):.1f} пикселей²")
    
    large_boxes = sum(1 for s in sizes if s > 10000)
    print(f"\n Боксов > 10000 пикселей²: {large_boxes} ({large_boxes/len(sizes)*100:.1f}%)")
    print("   Если много — уменьшите iou или добавьте в датасет примеров с плотными деревьями")

def show_few_shots(model, path):
    """path is a folder with .jpg files"""
    val_dir = Path(path)
    image_paths = list(val_dir.glob("*.jpg"))

    for filename in image_paths:
        results = model.predict(
            source=str(filename),
            conf=0.01,
            iou=0.25,
            imgsz=640
        )

        plotted_img = results[0].plot()

        plt.figure(figsize=(10, 7))
        plt.imshow(cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(filename.name)
        plt.show()
    
if __name__ == "__main__":
    model = YOLO(WEIGHTS_PATH)
    
    # best_conf, best_iou = find_best_conf_iou(model, PATH_TO_DATA)
    
    # analyze_box_sizes(model, "/home/jamradise/MegaDendrologAI/ML/Detection/Data/Tree-Quality-3-7/valid/images", best_conf, best_iou)

    show_few_shots(model, path="/home/jamradise/MegaDendrologAI/ML/Detection/Data/few-shots")