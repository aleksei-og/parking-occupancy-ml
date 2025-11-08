import json
import cv2
import os
import sys
from colorama import Fore, init

# Путь к папке с изображениями и JSON-файлу
IMAGES_DIR = r"D:\PKLot_Dataset\create data\train_7"  # ВНИМАТЕЛЬНО ВЫБИРАЙТЕ ПАПКУ
# не забываем, что ~~~~~_annotations разные для каждой папки
ANNOTATIONS_FILE = r"D:\PKLot_Dataset\create data\train_7\train_annotations.json"  # ВНИМАТЕЛЬНО ВЫБИРАЙТЕ ФАЙЛ

# Проверка существования папки с изображениями
if not os.path.exists(IMAGES_DIR):
    print(Fore.RED + f"Ошибка: Папка с изображениями не найдена:\n{IMAGES_DIR}")
    sys.exit(1)

# Проверка существования JSON-файла
if not os.path.exists(ANNOTATIONS_FILE):
    print(Fore.RED + f"Ошибка: JSON файл не найден:\n{ANNOTATIONS_FILE}")
    sys.exit(1)

print(Fore.GREEN + "Папка с изображениями и JSON файл найдены. Продолжаем выполнение...")
print("Управление: \nA <- \nB -> \nESC - выход")

# Загрузка аннотаций
with open(ANNOTATIONS_FILE, 'r') as f:
    coco_data = json.load(f)

# Преобразуем аннотации для быстрого доступа
annotations_by_image = {}
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    annotations_by_image.setdefault(img_id, []).append(ann)

# Сопоставление id → имя файла
images_by_id = {img['id']: img for img in coco_data['images']}
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

# Цвета для аннотаций
colors = {
    'space-empty': (0, 255, 0),  # зелёный
    'space-occupied': (0, 0, 255)  # красный
}

image_ids = sorted(images_by_id.keys())
index = 0
n = len(image_ids)

while True:
    img_id = image_ids[index]
    img_info = images_by_id[img_id]
    img_path = os.path.join(IMAGES_DIR, img_info['file_name'])

    img = cv2.imread(img_path)
    if img is None:
        print(f"Не удалось загрузить {img_path}")
        continue

    print("Индекс изображения - ", index)

    anns = annotations_by_image.get(img_id, [])
    for ann in anns:
        x, y, w, h = map(int, ann['bbox'])
        cat_name = categories[ann['category_id']]
        color = colors.get(cat_name, (255, 255, 255))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        # Текст рамок
        # cv2.putText(img, cat_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imshow('Parking Detection Viewer', img)

    key = cv2.waitKey(0)

    # Работает на английской раскладке
    if key == 27:  # Esc для выхода
        break
    elif key in [32, ord('d'), 83]:  # Пробел, D, → вперёд
        index = (index + 1) % n
    elif key in [ord('a'), 81]:  # A, ← назад
        index = (index - 1 + n) % n  # чтобы не уйти в -1


cv2.destroyAllWindows()
