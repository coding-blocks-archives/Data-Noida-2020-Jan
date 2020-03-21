import cv2

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import os

data = np.load("faces.npy")

X, y = data[:, :-1].astype(np.uint8), data[:, -1]

model = KNeighborsClassifier(4)
model.fit(X, y)

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:

    retval, image = cap.read()

    if retval:

        faces = classifier.detectMultiScale(image)

        if len(faces) > 0:

            faces = sorted(faces, key=lambda item: item[2] * item[3], reverse=True)
            face = faces[0]
            x, y, w, h = face

            cut = image[y:y+h, x:x+w]

            fix_cut = cv2.resize(cut, (100, 100))

            gray = cv2.cvtColor(fix_cut, cv2.COLOR_BGR2GRAY)

            result = model.predict(gray.reshape(1, -1))

            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 5)
            cv2.putText(image, str(result), (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

            cv2.imshow("my window", image)

    key = cv2.waitKey(10)

    if ord("q") == key:
        break
