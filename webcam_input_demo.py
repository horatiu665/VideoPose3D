import cv2
import time

video = cv2.VideoCapture(0)
a = 0

while True:
    a = a + 1

    # create frame object
    check, frame = video.read()

    print(check)
    print(frame)

    # grayscale??

    # show the frame
    cv2.imshow("Capturing", frame)

    # wait 1 millisec for a key
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

print(a)

video.release()

cv2.destroyAllWindows()
