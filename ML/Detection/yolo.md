# YOLO Results Documentation
## Overview

AI GENERATED

## Basic Usage

```python
from ultralytics import YOLO

# Загрузка модели
model = YOLO('yolo11n.pt')

# Детекция на изображении
results = model.predict(source='image.jpg')

# Получение первого результата (для одного изображения)
result = results[0]
```

## Results Object Structure

### Основные атрибуты объекта Results

| Атрибут | Тип | Описание |
|---------|-----|----------|
| `boxes` | `Boxes` | Объект с информацией о bounding boxes |
| `names` | `dict` | Словарь соответствия ID классов и их названий |
| `orig_img` | `np.ndarray` | Исходное изображение в формате numpy array |
| `orig_shape` | `tuple` | Размеры исходного изображения (height, width) |
| `path` | `str` | Путь к исходному файлу |
| `speed` | `dict` | Время выполнения разных этапов обработки |

### Пример доступа к атрибутам

```python
# Основная информация
print(f"Image path: {result.path}")
print(f"Image shape: {result.orig_shape}")
print(f"Class names: {result.names}")

# Информация о скорости обработки
print(f"Preprocess time: {result.speed['preprocess']}ms")
print(f"Inference time: {result.speed['inference']}ms")
print(f"Postprocess time: {result.speed['postprocess']}ms")
```

## Boxes Object

Объект `boxes` содержит всю информацию о обнаруженных объектах.

### Атрибуты Boxes

| Атрибут | Формат | Описание |
|---------|--------|----------|
| `xyxy` | `torch.Tensor` | Координаты bbox в формате [x1, y1, x2, y2] (абсолютные) |
| `xywh` | `torch.Tensor` | Координаты bbox в формате [x_center, y_center, width, height] |
| `xywhn` | `torch.Tensor` | Нормализованные координаты [0-1] |
| `conf` | `torch.Tensor` | Уровень уверенности детекции [0-1] |
| `cls` | `torch.Tensor` | ID классов обнаруженных объектов |

### Пример работы с Boxes

```python
if result.boxes is not None:
    boxes = result.boxes
    
    # Количество обнаруженных объектов
    num_detections = len(boxes)
    print(f"Detected objects: {num_detections}")
    
    # Доступ к данным первого обнаруженного объекта
    if num_detections > 0:
        # Координаты bounding box
        bbox_xyxy = boxes.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        bbox_xywh = boxes.xywh[0].cpu().numpy()  # [x_center, y_center, width, height]
        
        # Уверенность и класс
        confidence = boxes.conf[0].cpu().numpy()
        class_id = int(boxes.cls[0].cpu().numpy())
        class_name = result.names[class_id]
        
        print(f"Class: {class_name} (ID: {class_id})")
        print(f"Confidence: {confidence:.3f}")
        print(f"BBox XYXY: {bbox_xyxy}")
        print(f"BBox XYWH: {bbox_xywh}")
```

## Полный пример анализа результатов

```python
def analyze_yolo_results(results):
    """
    Детальный анализ результатов YOLO
    """
    for i, result in enumerate(results):
        print(f"\n{'='*50}")
        print(f"IMAGE {i+1}: {Path(result.path).name}")
        print(f"{'='*50}")
        
        print(f"Dimensions: {result.orig_shape}")
        print(f"Processing speed: {result.speed}")
        
        if result.boxes is None or len(result.boxes) == 0:
            print("No objects detected")
            continue
            
        boxes = result.boxes
        print(f"\nDetected objects: {len(boxes)}")
        print("-" * 30)
        
        # Сортируем по уверенности (от высокой к низкой)
        confidences = boxes.conf.cpu().numpy()
        sorted_indices = np.argsort(confidences)[::-1]
        
        for j, idx in enumerate(sorted_indices):
            box = boxes[idx]
            
            # Извлекаем данные
            xyxy = box.xyxy[0].cpu().numpy()
            xywh = box.xywh[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = result.names[class_id]
            
            # Рассчитываем площадь bounding box
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            
            print(f"Object {j+1}:")
            print(f"  Class: {class_name} (ID: {class_id})")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  BBox: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
            print(f"  Center: ({xywh[0]:.1f}, {xywh[1]:.1f})")
            print(f"  Size: {xywh[2]:.1f} × {xywh[3]:.1f}")
            print(f"  Area: {area:.1f} pixels")
            print()

# Использование
results = model.predict(source="image.jpg")
analyze_yolo_results(results)
```

## Форматы координат

### XYXY Format
```
[x_min, y_min, x_max, y_max]
```
- **x_min, y_min**: левый верхний угол
- **x_max, y_max**: правый нижний угол
- Координаты в пикселях относительно исходного изображения

### XYWH Format
```
[x_center, y_center, width, height]
```
- **x_center, y_center**: центр bounding box
- **width, height**: ширина и высота bbox
- Координаты в пикселях

### XYWHN Format (Normalized)
```
[x_center_norm, y_center_norm, width_norm, height_norm]
```
- Нормализованные значения [0-1] относительно размеров изображения

## Пример преобразования между форматами

```python
def convert_bbox_formats(xyxy, image_width, image_height):
    """
    Конвертация между разными форматами bounding boxes
    """
    x1, y1, x2, y2 = xyxy
    
    # XYXY to XYWH
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    xywh = [x_center, y_center, width, height]
    
    # XYWH to Normalized
    xywhn = [
        x_center / image_width,
        y_center / image_height, 
        width / image_width,
        height / image_height
    ]
    
    return {
        'xyxy': xyxy,
        'xywh': xywh,
        'xywhn': xywhn
    }

# Использование
bbox_data = convert_bbox_formats([100, 50, 300, 200], 640, 480)
```

## Обработка нескольких изображений

При обработке батча изображений, `results` содержит список объектов Results:

```python
# Обработка нескольких изображений
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = model.predict(source=image_paths)

for i, result in enumerate(results):
    print(f"Image {i+1}: {len(result.boxes) if result.boxes else 0} detections")
```

## Визуализация результатов

YOLO предоставляет встроенные методы для визуализации:

```python
# Сохранение изображений с детекциями
results = model.predict(source='image.jpg')
for i, result in enumerate(results):
    # Сохранение в файл
    result.save(filename=f'result_{i}.jpg')
    
    # Получение изображения с наложенными bbox
    plotted_image = result.plot()  # Возвращает numpy array
    
    # Отображение с помощью OpenCV
    import cv2
    cv2.imshow('Detection', plotted_image)
    cv2.waitKey(0)
```

## Практические примеры использования

### Фильтрация по уверенности
```python
def filter_detections_by_confidence(results, confidence_threshold=0.5):
    """Фильтрация детекций по порогу уверенности"""
    filtered_results = []
    
    for result in results:
        if result.boxes is None:
            filtered_results.append(result)
            continue
            
        # Фильтруем bounding boxes
        high_conf_indices = result.boxes.conf > confidence_threshold
        filtered_boxes = result.boxes[high_conf_indices]
        
        # Создаем новый объект Results с отфильтрованными данными
        result.boxes = filtered_boxes
        filtered_results.append(result)
    
    return filtered_results
```

### Экспорт результатов в JSON
```python
def results_to_json(results):
    """Конвертация результатов в JSON-формат"""
    json_results = []
    
    for result in results:
        result_data = {
            'image_path': result.path,
            'image_size': result.orig_shape,
            'detections': []
        }
        
        if result.boxes is not None:
            for box in result.boxes:
                detection = {
                    'class_id': int(box.cls[0].cpu().numpy()),
                    'class_name': result.names[int(box.cls[0].cpu().numpy())],
                    'confidence': float(box.conf[0].cpu().numpy()),
                    'bbox_xyxy': box.xyxy[0].cpu().numpy().tolist(),
                    'bbox_xywh': box.xywh[0].cpu().numpy().tolist()
                }
                result_data['detections'].append(detection)
        
        json_results.append(result_data)
    
    return json_results
```

## Важные замечания

1. **Тензоры находятся на GPU**: Данные в атрибутах boxes изначально находятся на GPU, используйте `.cpu().numpy()` для конвертации.

2. **Проверка на наличие детекций**: Всегда проверяйте `if result.boxes is not None` перед работой с bounding boxes.

3. **Нормализация координат**: Координаты в `xywhn` нормализованы относительно размеров изображения.

4. **Множественные изображения**: При обработке нескольких изображений, `results` является списком объектов Results.

Этот документ покрывает основные аспекты работы с результатами YOLO. Для более детальной информации обращайтесь к официальной документации Ultralytics YOLO.