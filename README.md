
# Parking Occupancy Detection with ML

## Описание проекта

Система, определяющая занятость парковочных мест на основе изображений с камер наблюдения. Применяется компьютерное зрение и машинное обучение для детекции и классификации мест.

Проект разработан в рамках курса по ML в НИТУ МИСИС. Команда из 4 человек, каждый внёс вклад в ключевые этапы пайплайна.


## Цель проекта

Автоматизировать мониторинг парковок с помощью CV/ML:
- детекция парковочных мест
- определение их статуса (занято/свободно)
- визуализация и счётчик свободных мест


## Структура проекта

- `Ml project.ipynb` — основной ноутбук с пайплайном
- `create_parking_dataset.py` — скрипт подготовки датасета
- `create_json_sample.py` — генерация примеров аннотаций
- `create_mask.py` — создание масок для изображений
- `learning_SVM_HOG.py` — бейзлайн на основе SVM + HOG
- `learning_log_loss.py` — логистическая регрессия
- `split_dataset.py` — разделение датасета на train/test
- `inference.py` — скрипт инференса: предсказание занятости по кадрам (YOLO и CNN)
- `models_arch.py` — архитектуры моделей: `SmallCNN`, `BetterCNNWithResiduals`
- `video_processor.py` — обработка видео: применение модели, визуализация, генерация выходного видео
- `main.ipynb` — финальный ноутбук с полной цепочкой обработки
- `config3.yaml` — конфигурационный файл с параметрами запуска
- `bmodel3_complete.pth` — обученная модель CNN
- `best.pt` — весы модели YOLOv8
- `input.mp4` — входное видео для обработки


## Данные

Использован датасет [PKLot](https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset):  
12 000+ изображений парковок при разных условиях.

Использованы аннотации в формате COCO: `space-empty` / `space-occupied`.


## Используемые технологии

- Python 3.10
- OpenCV, YOLOv8, NumPy, scikit-learn, Matplotlib
- Jupyter Notebook
- Сегментация, классификация, SVM, HOG


## Метрики

- **Accuracy**: базовая метрика классификации
- **F1-score**: баланс точности и полноты


## Результаты

| Модель                   | Accuracy (test) | F1-мера (test) | Precision | Recall |
| ------------------------ | --------------- | -------------- | --------- | ------ |
| **Logistic Regression**  | 0.83            | 0.82           | 0.81      | 0.83   |
| **SVM + HOG**            | 0.83            | 0.82           | 0.81      | 0.83   |
| **PCA + Random Forest**  | 0.87            | 0.86           | 0.79      | 0.98   |
| **Simple CNN**           | 0.97            | 0.97           | 0.98      | 0.94   |
| **Simple CNN (CNRPark)** | 0.94            | 0.93           | 0.92      | 0.94   |


## Выводы

- Несмотря на высокие метрики, модель не выполняет автоматическую детекцию парковочных мест.
- Алгоритм работает только при наличии заранее размеченных данных (координат мест из JSON) и изображения/видео.
- Модель может ошибочно определять место как занятое, если на нём находится человек или кто-то проходит мимо.
- Тем не менее, система показывает устойчивые результаты при классификации вырезанных мест в условиях шума, теней и перспективных искажений.
- Возможности масштабирования: детекция мест, улучшение устойчивости к шуму, предсказание загруженности, применение к видео в реальном времени.


## Команда

- Камышева Ольга — сбор данных, предобработка
- Оганов Алексей — детекция парковочных мест
- Калмыкова Мария — классификация
- Кудашов Юрий — визуализация, финальный пайплайн


## Литература

- [PKLot Dataset on Kaggle](https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset)
- Mazerov S. (2023). [Определение свободного парковочного места с помощью Computer Vision (Habr)](https://habr.com/ru/articles/738720/)
- Amato G. et al. (2016). *Car parking occupancy detection using smart camera networks and Deep Learning.* [PDF](https://falchi.isti.cnr.it/Draft/2016-ISCC-Draft.pdf)
- Yuldashev Y. et al. (2023). *Parking Lot Occupancy Detection with Improved MobileNetV3.* [ResearchGate](https://www.researchgate.net/publication/373667077_Parking_Lot_Occupancy_Detection_with_Improved_MobileNetV3)
- Belousov P. (2019). [Я решил проблему с парковкой у дома с помощью машинного обучения (vc.ru)](https://vc.ru/dev/57372-ya-reshil-problemu-s-parkovkoi-u-doma-s-pomoshyu-mashinnogo-obucheniya)
- Priyanka Kumari. (2024). [Building Parking Space Detection System using PyTorch and Super Gradients (Labellerr)](https://www.labellerr.com/blog/building-parking-space-detection-system-pytorch-super-gradients/)
- Лаптев В.В. и др. (2022). *Анализ парковочного пространства посредством компьютерного зрения.* [GraphiCon PDF](https://www.graphicon.ru/html/2022/papers/paper_052.pdf)
