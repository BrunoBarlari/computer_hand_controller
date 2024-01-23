import cv2
import numpy as np


video_capture = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")


# Face detection
def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return img


while True:
    ret, frame = video_capture.read()
    frame = detect_faces(frame)
    cv2.imshow("frame", frame)  # Display the frame

    # Press q to quit
    if cv2.waitKey(1) == ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()
