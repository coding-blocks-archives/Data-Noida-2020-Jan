import cv2

import numpy as np

import os

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

name = input("Enter your name : ")
img_count = int(input("Number of images : "))

train_x = []

while True:

    retval, image = cap.read()

    gray = None

    if retval:

        faces = classifier.detectMultiScale(image)

        print(faces)

        if len(faces) > 0:

            faces = sorted(faces, key=lambda item: item[2] * item[3], reverse=True)
            face = faces[0]
            x, y, w, h = face

            cut = image[y:y+h, x:x+w]

            fix_cut = cv2.resize(cut, (100, 100))

            gray = cv2.cvtColor(fix_cut, cv2.COLOR_BGR2GRAY)

            cv2.imshow("my window", gray)

    key = cv2.waitKey(100)

    if ord("q") == key:
        break
    elif ord("c") == key:
        if retval:
            train_x.append(gray.flatten())
            print(img_count - len(train_x), "to go")

            if img_count == len(train_x):
                break

X = np.array(train_x)
y = np.full((len(X), 1), name)

data = np.hstack([X, y])

if os.path.exists("faces.npy"):
    old = np.load("faces.npy")
    data = np.vstack([old, data])

np.save("faces.npy", data)

cap.release()
cv2.destroyAllWindows()