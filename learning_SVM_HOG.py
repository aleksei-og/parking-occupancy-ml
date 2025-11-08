import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier  # Линейный SVM через SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib


#  Загрузка данных
def load_images(folder, label):
    images = []
    labels = []
    filenames = []
    for filename in tqdm(os.listdir(folder), desc=f'Загрузка {os.path.basename(folder)}'):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Размер для HOG
            images.append(img)
            labels.append(label)
            filenames.append(img_path)
    return images, labels, filenames


occupied_dir = "C:/PKLot_Dataset/create data/tmptest2/occupied_2906"
empty_dir = "C:/PKLot_Dataset/create data/tmptest2/empty_2906"

X_occupied, y_occupied, occupied_files = load_images(occupied_dir, label=1)
X_empty, y_empty, empty_files = load_images(empty_dir, label=0)

X = np.array(X_occupied + X_empty)
y = np.array(y_occupied + y_empty)
all_files = occupied_files + empty_files


# Извлечение HOG-признаков
def extract_hog_features(images):
    features = []
    for img in tqdm(images, desc='Извлечение HOG'):
        hog_feature = hog(
            img,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            channel_axis=None
        )
        features.append(hog_feature)
    return np.array(features)


X_hog = extract_hog_features(X)

# 3. Разделение данных
X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
    X_hog, y, all_files, test_size=0.2, random_state=42
)

#  Стандартизация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. Обучение SVM с графиками
def train_svm_with_metrics(X_train, y_train, X_val, y_val, n_epochs=10):
    """Обучение SVM с сохранением метрик на каждой эпохе."""
    model = SGDClassifier(
        loss='hinge',  # hinge loss = линейный SVM
        learning_rate='optimal',
        eta0=0.01,
        random_state=42
    )

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in tqdm(range(n_epochs), desc='Обучение SVM'):
        model.partial_fit(X_train, y_train, classes=np.unique(y_train))

        # Расчет потерь и точности на тренировочных данных
        train_acc.append(accuracy_score(y_train, model.predict(X_train)))
        train_loss.append(np.mean(np.maximum(0, 1 - y_train * model.decision_function(X_train))))  # Hinge loss

        # Расчет на валидации
        val_acc.append(accuracy_score(y_val, model.predict(X_val)))
        val_loss.append(np.mean(np.maximum(0, 1 - y_val * model.decision_function(X_val))))

    return model, train_loss, val_loss, train_acc, val_acc


model, train_loss, val_loss, train_acc, val_acc = train_svm_with_metrics(
    X_train_scaled, y_train, X_test_scaled, y_test
)

# Графики обучения
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train')
plt.plot(val_loss, label='Validation')
plt.title('Hinge Loss (потери)')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Train')
plt.plot(val_acc, label='Validation')
plt.title('Точность')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()

#  Оценка модели
y_pred = model.predict(X_test_scaled)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nОтчет по классам:")
print(classification_report(y_test, y_pred))

#  Визуализация ошибок
errors = np.where(y_pred != y_test)[0]

plt.figure(figsize=(15, 8))
for i, idx in enumerate(errors[:6]):
    img = cv2.imread(files_test[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
    plt.axis('off')
plt.suptitle("Примеры ошибок классификации", fontsize=16)
plt.show()

# # Сохранение модели
# joblib.dump(model, 'svm_hog_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# print("\nМодель сохранена: 'svm_hog_model.pkl'")