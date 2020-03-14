import cv2

cap = cv2.VideoCapture(0)

while True:
    retval, image = cap.read()

    print(type(image))

    if retval:
        cv2.imshow("my photo", image)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break
    if key == ord("c"):
        cv2.imwrite("classroom.jpg", image)


cap.release()
cv2.destroyAllWindows()