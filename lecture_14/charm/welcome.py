import cv2

cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:

    retval, image = cap.read()

    if retval:

        faces = classifier.detectMultiScale(image)

        print(faces)

        if len(faces) > 0:
            face = faces[0]
            x, y, w, h = face

            cut = image[y:y+h, x:x+w]

            fix_cut = cv2.resize(cut, (500, 500))

            gray = cv2.cvtColor(fix_cut, cv2.COLOR_BGR2GRAY)

            cv2.imshow("my window", gray)

    key = cv2.waitKey(100)

    if ord("b") == key:
        break

cap.release()
cv2.destroyAllWindows()