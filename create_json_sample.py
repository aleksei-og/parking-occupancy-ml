"""
как же я устал с этим скриптом...
"""

import json
import os
import sys
from collections import defaultdict
from colorama import Fore, init

init(autoreset=True)


def create_accurate_annotations(image_dir, output_json_name, source_jsons):
    """Создает точные аннотации без дубликатов и перекрестных совпадений"""
    print(Fore.CYAN + f"\nСоздание {output_json_name} для {image_dir}")

    # 1. Получаем список файлов без полных путей
    image_files = {os.path.basename(f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))}

    if not image_files:
        print(Fore.RED + "Ошибка: В папке нет изображений!")
        return None

    # 2. Инициализация структур данных
    output = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "space-empty", "supercategory": "spaces"},
            {"id": 2, "name": "space-occupied", "supercategory": "spaces"}
        ]
    }

    # 3. Временные хранилища
    file_to_new_id = {}  # {filename: new_image_id}
    bbox_registry = defaultdict(dict)  # {image_id: {bbox_key: category_id}}
    source_counts = defaultdict(int)  # Для отладки

    # 4. Сначала собираем все изображения
    new_image_id = 1
    for json_path in source_jsons:
        with open(json_path) as f:
            data = json.load(f)

        for img in data["images"]:
            filename = os.path.basename(img["file_name"])
            if filename in image_files and filename not in file_to_new_id:
                output["images"].append({
                    "id": new_image_id,
                    "file_name": filename,
                    "width": img["width"],
                    "height": img["height"],
                    "original_source": os.path.basename(json_path)
                })
                file_to_new_id[filename] = new_image_id
                new_image_id += 1

    # 5. Теперь собираем аннотации с проверкой
    for json_path in source_jsons:
        with open(json_path) as f:
            data = json.load(f)

        # Создаем mapping старых ID к новым
        id_mapping = {img["id"]: file_to_new_id[os.path.basename(img["file_name"])]
                      for img in data["images"]
                      if os.path.basename(img["file_name"]) in file_to_new_id}

        for ann in data["annotations"]:
            if ann["image_id"] in id_mapping:
                new_image_id = id_mapping[ann["image_id"]]
                bbox_key = tuple(round(x, 1) for x in ann["bbox"])  # Округление для избежания float ошибок

                # Если bbox уже существует, выбираем наиболее вероятную категорию
                if bbox_key not in bbox_registry[new_image_id]:
                    bbox_registry[new_image_id][bbox_key] = ann["category_id"]
                    source_counts[os.path.basename(json_path)] += 1

    # 6. Формируем финальные аннотации
    annotation_id = 1
    for image_id, bboxes in bbox_registry.items():
        for bbox, cat_id in bboxes.items():
            output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": list(bbox),
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            annotation_id += 1

    # 7. Валидация
    print(f"\nРезультаты:")
    print(f"- Изображений: {len(output['images'])}")
    print(f"- Аннотаций: {len(output['annotations'])}")
    print(f"- Распределение категорий:")
    cat_stats = defaultdict(int)
    for ann in output["annotations"]:
        cat_stats[ann["category_id"]] += 1
    for cat in output["categories"]:
        print(f"  {cat['name']}: {cat_stats.get(cat['id'], 0)}")

    print("\nИсточники аннотаций:")
    for src, count in source_counts.items():
        print(f"  {src}: {count} аннотаций")

    # 8. Сохранение
    output_path = os.path.join(image_dir, output_json_name)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(Fore.GREEN + f"\n✅ Файл {output_json_name} успешно создан")
    return output_path


# Конфигурация для всех путей
configs = [
    {
        "image_dir": r"D:\PKLot_Dataset\create data\test_1",
        "output_json_name": "test_annotations.json",
        "source_jsons": [
            r"D:\PKLot_Dataset\train\_annotations.coco.json",
            r"D:\PKLot_Dataset\valid\_annotations.coco.json",
            r"D:\PKLot_Dataset\test\_annotations.coco.json"
        ]
    },
    {
        "image_dir": r"D:\PKLot_Dataset\create data\train_7",
        "output_json_name": "train_annotations.json",
        "source_jsons": [
            r"D:\PKLot_Dataset\train\_annotations.coco.json",
            r"D:\PKLot_Dataset\valid\_annotations.coco.json",
            r"D:\PKLot_Dataset\test\_annotations.coco.json"
        ]
    },
    {
        "image_dir": r"D:\PKLot_Dataset\create data\valid_2",
        "output_json_name": "valid_annotations.json",
        "source_jsons": [
            r"D:\PKLot_Dataset\train\_annotations.coco.json",
            r"D:\PKLot_Dataset\valid\_annotations.coco.json",
            r"D:\PKLot_Dataset\test\_annotations.coco.json"
        ]
    }
]


if __name__ == "__main__":
    # Создаем аннотации для всех конфигураций
    for config in configs:
        # Проверка существования папки с изображениями
        if not os.path.exists(config['image_dir']):
            print(Fore.RED + f"Ошибка: Папка с изображениями не найдена:\n{config['image_dir']}")
            sys.exit(1)
        print(Fore.YELLOW + f"\n{'=' * 50}")
        print(Fore.YELLOW + f"Обработка: {config['image_dir']}")
        create_accurate_annotations(**config)

    print(Fore.GREEN + "\nВсе аннотации успешно созданы!")
