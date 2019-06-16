from urllib.request import urlopen
from ssl import SSLContext, PROTOCOL_TLSv1
import numpy as np
import cv2

# Для определения объекта в видео потоке используя каскад Хаара
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# На каждого человека - один айди
face_id = 1

# Счетчик изображений лиц
count = 0

cam = cv2.VideoCapture(0)

while (True):

    ret, im = cam.read()

    # Получение видеофрейма
    # info = urlopen(url, context=gcontext).read()

    # imgNp = np.array(bytearray(im), dtype=np.uint8)
    # image_frame = cv2.imdecode(imgNp, -1)

    # Конвертация в грейскейл
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Определяем фреймы разных размеров
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Подстановка изображения в прямоугольник
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

        count += 1

        # Сохранение изображения в папку датасета
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

        print(count)

        # Отобразить прямоугольник вокруг лица
        cv2.imshow('frame', im)

    k = cv2.waitKey(33)
    if k == 27:  # Esc для остановки
        break

    # Если изображений больше 100 - остановка
    elif count > 100:
        break
