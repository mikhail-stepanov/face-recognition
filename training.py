import cv2, os
import numpy as np
from PIL import Image

recognizer = cv2.cv2.face.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");


# Метод получения изображения и подписи
def getImagesAndLabels(path):
    # Получение пути файлов
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    faceSamples = []

    ids = []

    for imagePath in imagePaths:

        # Конвертация в грейскейл
        PIL_img = Image.open(imagePath).convert('L')

        # PIL в нампай массив
        img_numpy = np.array(PIL_img, 'uint8')

        # Получение id изображения
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # Получение лица из датасета
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            # Добавление изображения в лица
            faceSamples.append(img_numpy[y:y + h, x:x + w])

            # Тоже для айди
            ids.append(id)

    return faceSamples, ids


# Получение всех лиц и айи
faces, ids = getImagesAndLabels('dataset')

# Обучение модели
recognizer.train(faces, np.array(ids))

# Сохранение модели в формате yml
recognizer.write('trainer/trainer.yml')
