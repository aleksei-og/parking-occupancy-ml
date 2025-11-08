"""
Тут я создаю выборку для тренировки нашей модели,
чтобы нашему железу было не сильно тяжело

Моя папка start - копия всех изображений с нужной камеры (копировал вручную)
из исходных файлов: train, valid, test
"""

import os
import sys
import shutil
import random
from colorama import Fore, init


# Путь к папке с изображениями
start_dir = r"D:\PKLot_Dataset\create data\start"
# Путь к папке в которой созадим папки: train, valid, test
base_dir = r"D:\PKLot_Dataset\create data"


# Получаем список всех файлов в start
all_files = [f for f in os.listdir(start_dir) if f.lower().endswith('.jpg')]


# Инициализация colorama
init(autoreset=True)

# Кол-во нужных файлов
while True:
    try:
        print(Fore.YELLOW + "Пожалуйста, введите общее кол-во изображений не больше" + Fore.RED +
              " {}: ".format(len(all_files)), end="")
        total_needed = float(input())

        if total_needed > len(all_files):
            print(Fore.RED + "Ошибка: введённое число превышает доступное количество изображений!")
        elif total_needed <= 0:
            print(Fore.RED + "Ошибка: число должно быть больше нуля!")
        else:
            break
    except ValueError:
        print(Fore.RED + "Ошибка: пожалуйста, введите число.")

train_count = int(total_needed * 0.7)
valid_count = int(total_needed * 0.2)
test_count = int(total_needed * 0.1)

total_needed = int(total_needed)

# Папки назначения
train_dir = os.path.join(base_dir, "train_{}".format(train_count))
valid_dir = os.path.join(base_dir, "valid_{}".format(valid_count))
test_dir = os.path.join(base_dir, "test_{}".format(test_count))

# Создание папок, если их нет
for folder in [train_dir, valid_dir, test_dir]:
    if os.path.exists(folder):
        print(Fore.YELLOW + f"\nПапки уже существуют.")
        choice = input(Fore.CYAN + "Хотите перезаписать их? (y/n): ").strip().lower()
        if choice == 'y':
            shutil.rmtree(folder)  # Удаление папки со всем содержимым
            os.makedirs(folder)
            print(Fore.GREEN + f"Папка '{folder}' была перезаписана.")
        else:
            print(Fore.RED + "\nЗавершение выполнения.")
            sys.exit(1)
    else:
        os.makedirs(folder)
        print(Fore.GREEN + f"Папка '{folder}' успешно создана.")


# Перемешиваем и выбираем total_needed случайных
selected_files = random.sample(all_files, total_needed)

# Перемешиваем заново для разбиения
random.shuffle(selected_files)

# Разбиваем
train_files = selected_files[:train_count]
valid_files = selected_files[train_count:train_count + valid_count]
test_files = selected_files[train_count + valid_count:]


# Функция копирования
def copy_files(file_list, destination):
    for file_name in file_list:
        src_path = os.path.join(start_dir, file_name)
        dst_path = os.path.join(destination, file_name)
        shutil.copy2(src_path, dst_path)


# Копируем
copy_files(train_files, train_dir)
copy_files(valid_files, valid_dir)
copy_files(test_files, test_dir)

print(Fore.GREEN + "Выборка создана! Файлы успешно распределены по папкам: " + Fore.YELLOW + "train, valid, test.")
