import cv2
# import numpy as np
# from ssl import SSLContext, PROTOCOL_TLSv1
# from urllib.request import urlopen

# Создаем локальный паттерн для распознавания лица
recognizer = cv2.cv2.face.LBPHFaceRecognizer_create()

# Загружаем натреннерованную модель
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"

# Создаем классификатор для модели
faceCascade = cv2.CascadeClassifier(cascadePath);

# Выставляет стиль текста
font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)

while True:
    # Получаем фрейм видео из потока
    # gcontext = SSLContext(PROTOCOL_TLSv1)
    # info = urlopen(url, context=gcontext).read()

    ret, im = cam.read()

    # imgNp = np.array(bytearray(info), dtype=np.uint8)
    # im = cv2.imdecode(imgNp, -1)

    # Переводим фрейм в грейскейл
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Получаем все лица из видеофрейма
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # Создаем прямоугольник вокруг лица
        cv2.rectangle(im, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)

        # Определяем принадлежность лица к ID
        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

        # Если ID существует
        if (Id == 1):
            Id = "Mikhail"
        # # Убрать коммент, в случае если распознаются другие лица и заменить ID из датасета
        elif (Id == 2):
            Id = "Mikhail"  # Имя другой персоны
        elif (Id == 2):
            Id = ""
        else:
            Id = "Uknown"

        # Добавить описание "кто это" на картинку
        cv2.rectangle(im, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
        cv2.putText(im, str(Id), (x, y - 40), font, 2, (255, 255, 255), 3)

    # Отобразить картинку с прямоугольником по границе
    cv2.imshow('im', im)

    # Закрыть прогу тыкнув q
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
