import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib


def load_images(folder, label):
    # Загрузка изображений с проверкой ошибок
    images = []
    labels = []
    filenames = []

    if not os.path.exists(folder):
        print(f"Ошибка: папка {folder} не существует!")
        return np.array([]), np.array([]), []

    files = os.listdir(folder)
    if not files:
        print(f"Папка {folder} пуста!")
        return np.array([]), np.array([]), []

    for filename in tqdm(files, desc=f'Загрузка {os.path.basename(folder)}'):
        try:
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"\nНе удалось загрузить {filename}")
                continue

            img = cv2.resize(img, (32, 32))
            images.append(img.flatten())
            labels.append(label)
            filenames.append(img_path)

        except Exception as e:
            print(f"\nОшибка при обработке {filename}: {str(e)}")
            continue

    if not images:
        print(f"Не загружено ни одного изображения из {folder}!")
        return np.array([]), np.array([]), []

    return np.array(images), np.array(labels), filenames


# Пути к данным
occupied_dir = "C:/PKLot_Dataset/create data/tmptest2/occupied_2906"
empty_dir = "C:/PKLot_Dataset/create data/tmptest2/empty_2906"

print("Начало загрузки данных...")
X_occupied, y_occupied, occupied_files = load_images(occupied_dir, label=1)
X_empty, y_empty, empty_files = load_images(empty_dir, label=0)

# Проверка данных
if len(X_occupied) == 0 or len(X_empty) == 0:
    print("Ошибка: не удалось загрузить данные. Проверьте пути и содержимое папок.")
    exit()

X = np.vstack((X_occupied, X_empty))
y = np.concatenate((y_occupied, y_empty))
all_files = occupied_files + empty_files

print(f"\nУспешно загружено: {len(y)} изображений")

# Разделение данных
X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
    X, y, all_files, test_size=0.2, random_state=42
)

# Стандартизация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA(n_components=50, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# Обучение модели
def train_model(X_train, y_train, X_val, y_val, n_epochs=50):
    model = SGDClassifier(loss='log_loss', learning_rate='adaptive',
                          eta0=0.01, random_state=42)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in tqdm(range(n_epochs), desc='Обучение модели'):
        model.partial_fit(X_train, y_train, classes=np.unique(y_train))

        # Метрики для train
        train_prob = model.predict_proba(X_train)
        train_loss.append(log_loss(y_train, train_prob))
        train_acc.append(accuracy_score(y_train, model.predict(X_train)))

        # Метрики для validation
        val_prob = model.predict_proba(X_val)
        val_loss.append(log_loss(y_val, val_prob))
        val_acc.append(accuracy_score(y_val, model.predict(X_val)))

    return model, train_loss, val_loss, train_acc, val_acc


model, train_loss, val_loss, train_acc, val_acc = train_model(
    X_train_pca, y_train, X_test_pca, y_test
)

# Графики обучения
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train')
plt.plot(val_loss, label='Validation')
plt.title('Функция потерь')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Train')
plt.plot(val_acc, label='Validation')
plt.title('Точность')
plt.legend()
plt.show()

# Оценка модели
y_pred = model.predict(X_test_pca)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Визуализация ошибок
errors = np.where(y_pred != y_test)[0]
plt.figure(figsize=(15, 8))
for i, idx in enumerate(errors[:6]):
    img = cv2.imread(files_test[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
    plt.axis('off')
plt.suptitle("Примеры ошибок классификации")
plt.show()

# # Сохранение моделей
# joblib.dump(pca, 'pca_model.pkl')
# joblib.dump(model, 'sgd_model.pkl')
# print("\nМодели сохранены в pca_model.pkl и sgd_model.pkl")