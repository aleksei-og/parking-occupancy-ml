import os
import json
import cv2
import sys
import shutil
from tqdm import tqdm
from pathlib import Path
from colorama import Fore, init

init(autoreset=True)


def create_dataset(annotations_path, output_root, number_dataset):
    """Создает датасет с разделением на занятые/свободные места"""
    # Загружаем аннотации
    with open(annotations_path) as f:
        data = json.load(f)

    # Создаем словарь для быстрого доступа к изображениям
    images = {img['id']: img for img in data['images']}

    # Обрабатываем все аннотации
    for ann in tqdm(data['annotations'], desc='Processing annotations'):
        img_info = images[ann['image_id']]
        img_path = os.path.join(os.path.dirname(annotations_path), img_info['file_name'])

        # Загружаем изображение
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Вырезаем ROI (Region of Interest)
        x, y, w, h = map(int, ann['bbox'])
        roi = img[y:y + h, x:x + w]

        # Определяем путь для сохранения
        if ann['category_id'] == 1:  # space-empty
            save_dir = os.path.join(output_root, f'empty_{number_dataset}')
        elif ann['category_id'] == 2:  # space-occupied
            save_dir = os.path.join(output_root, f'occupied_{number_dataset}')
        else:
            continue

        # Сохраняем вырезанное изображение
        base_name = f"{Path(img_info['file_name']).stem}_{ann['id']}.jpg"
        save_path = os.path.join(save_dir, base_name)
        cv2.imwrite(save_path, roi)


if __name__ == "__main__":
    # Конфигурация
    ANNOTATIONS_PATH = r"D:\PKLot_Dataset\create data\train_700\train_annotations.json"
    OUTPUT_ROOT = r"D:\PKLot_Dataset\create data\tmptest"

    # Получаем номер датасета из пути
    number_dataset = ANNOTATIONS_PATH.split('\\')[-2].split('_')[-1]

    # Определяем пути к папкам
    folder_occupied = os.path.join(OUTPUT_ROOT, f'occupied_{number_dataset}')
    folder_empty = os.path.join(OUTPUT_ROOT, f'empty_{number_dataset}')

    # Проверка на наличие папок
    if os.path.exists(folder_occupied) or os.path.exists(folder_empty):
        print(Fore.YELLOW + f"Папки уже существуют.")
        choice = input(Fore.CYAN + "Хотите перезаписать их? (y/n): ").strip().lower()
        if choice == 'y':
            if os.path.exists(folder_occupied):
                shutil.rmtree(folder_occupied)
            if os.path.exists(folder_empty):
                shutil.rmtree(folder_empty)
            os.makedirs(folder_occupied)
            os.makedirs(folder_empty)
            print(Fore.GREEN + f"Папки были перезаписаны.")
        else:
            print(Fore.RED + "\nЗавершение выполнения.")
            sys.exit(1)
    else:
        os.makedirs(folder_occupied)
        os.makedirs(folder_empty)
        print(Fore.GREEN + f"Папки были успешно созданы.")

    # Создаем датасет
    create_dataset(ANNOTATIONS_PATH, OUTPUT_ROOT, number_dataset)
    print(Fore.GREEN + f"\n✅ Датасет успешно создан!" + Fore.YELLOW + f"\nПуть к датасету: {OUTPUT_ROOT}")
