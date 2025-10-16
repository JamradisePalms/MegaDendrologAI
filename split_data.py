import json
import numpy as np
import os
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def augment_images_for_minority_classes(data, target_count=10, output_dir="augmented_images"):
    """
    Добавляет аугментированные изображения для классов с количеством изображений меньше target_count
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    class_counts = Counter([item['tree_type'] for item in data])

    transform = transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.Rotate(limit=15, p=0.4),
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            A.ToGray(p=0.7),
        ], p=0.8),
        A.OneOf([
            A.GaussianBlur(blur_limit=(1, 3)),
            A.GaussNoise(var_limit=(10, 50)),
            A.MotionBlur(blur_limit=3),
        ], p=0.5),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
    ])
    
    augmented_data = []
    
    for class_name, count in class_counts.items():
        if count < target_count and class_name:
            print(f"Аугментируем класс '{class_name}' (всего {count} изображений)")
            
            class_items = [item for item in data if item['tree_type'] == class_name]
            needed_count = target_count - count
            
            successful_augmentations = 0
            
            for i in range(needed_count):
                original_item = class_items[i % len(class_items)]
                image_path = Path(original_item["image"])
                
                image = None
                
                try:
                    from PIL import Image
                    pil_image = Image.open(image_path)
                    image = np.array(pil_image)
                    if len(image.shape) == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    elif image.shape[2] == 4:
                        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                except Exception as e:
                    print(f"Ошибка PIL для {image_path}: {e}")
                    continue
                
                try:
                    augmented = transform(image=image)
                    augmented_image = augmented["image"]
                    
                    aug_filename = f"{image_path.stem}_aug_{successful_augmentations}{image_path.suffix}"
                    aug_path = Path(output_dir) / aug_filename
                    
                    pil_augmented = Image.fromarray(augmented_image)
                    pil_augmented.save(str(aug_path))
                    
                    aug_item = original_item.copy()
                    aug_item["image"] = str(aug_path)
                    aug_item["is_augmented"] = True 
                    
                    augmented_data.append(aug_item)
                    successful_augmentations += 1
                    
                    print(f"Создано аугментированное изображение: {aug_filename}")
                    
                except Exception as e:
                    print(f"Ошибка при аугментации {image_path}: {e}")
                    continue
    data.extend(augmented_data)
    
    class_counts_after = Counter([item['tree_type'] for item in data])
    print("\nРаспределение классов после аугментации:")
    for class_name, count in class_counts_after.items():
        print(f"  {class_name}: {count} изображений")
    
    print(f"Добавлено {len(augmented_data)} аугментированных изображений")
    return data

def main():
    json_path = r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Classification\new_data\combined_data.json"
    images_dir = r"C:\Users\shari\OneDrive\Рабочий стол\Hack-processed-data\tree_crops"
    output_dir = r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Classification\augmented_images"

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f, )
    
    labels = [item['tree_type'] for item in data]
    
    train_data, valid_data = train_test_split(
        data, 
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=labels,
    )
    
    def find_image_files(data_list, search_dir):
        found_data = []
        missing_data = []
        
        for item in tqdm(data_list, desc="Поиск изображений"):
            image_path = Path(item["image"])
            
            if image_path.exists():
                found_data.append(item)
                continue
                
            image_suffix = image_path.name
            found_files = list(Path(search_dir).rglob(image_suffix))
            
            if found_files:
                item["image"] = str(found_files[0])
                found_data.append(item)
            else:
                missing_data.append(item)
                
        return found_data, missing_data
    
    train_found, train_missing = find_image_files(train_data, images_dir)
    valid_found, valid_missing = find_image_files(valid_data, images_dir)
    
    # train_found = augment_images_for_minority_classes(train_found, target_count=30, output_dir=output_dir)

    with open(r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Classification\new_data\train_data.json", 'w', encoding='utf-8') as f:
        json.dump(train_found, f, ensure_ascii=False, indent=4)
    
    with open(r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Classification\new_data\valid_data.json", 'w', encoding='utf-8') as f:
        json.dump(valid_found, f, ensure_ascii=False, indent=4)
    
    with open('missing_files.json', 'w', encoding='utf-8') as f:
        json.dump({
            'train_missing': train_missing,
            'valid_missing': valid_missing
        }, f, ensure_ascii=False, indent=4)
    
    print(f"\n--- Результаты обработки ---")
    print(f"Всего записей: {len(data)}")
    print(f"Train выборка: {len(train_found)} записей")
    print(f"Valid выборка: {len(valid_found)} записей")
    print(f"Пропущено в train: {len(train_missing)} файлов")
    print(f"Пропущено в valid: {len(valid_missing)} файлов")
    print(f"Общее количество пропущенных файлов: {len(train_missing) + len(valid_missing)}")

if __name__ == "__main__":
    main()