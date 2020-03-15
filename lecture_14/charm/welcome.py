import cv2

cap = cv2.VideoCapture(0)

while True:

    retval, image = cap.read()

    if retval:
        # cv2.imshow("my window", image)

        cut = image[:200, :200]
        cv2.imshow("my window", cut)

    key = cv2.waitKey(1)

    if ord("b") == key:
        break

cap.release()
cv2.destroyAllWindows()