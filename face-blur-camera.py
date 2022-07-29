from cv2 import blur
import numpy as np
import tensorflow as tf
import time
import cv2
import sys

'''
# 이미지로 모자이크 확인
path = 'C:/Users/USER/OneDrive - 주식회사 인티그리트/문서/GitHub/Face-Verification-Project/11.jpg'
img_array = np.fromfile(path, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

if img is None:
    print('Image load failed')
    sys.exit()

face_detector = cv2.CascadeClassifier('haarcascade_frontface.xml')

face = face_detector.detectMultiScale(img)

for (x, y, w, h) in face:
    # face_box = cv2.rectangle(img, (x,y,w,h), (0,0,255),2)
    faces = img[y:y+h, x:x+w]
    image_blur = cv2.blur(faces,(50,50))
    img_blur = img
    img_blur[y:y+h, x:x+w] = image_blur
    cv2.imshow('image', img_blur)

cv2.waitKey()
cv2.destroyAllWindows()
'''

# 카메라 블러

cap = cv2.VideoCapture(1)

face_detector = cv2.CascadeClassifier('haarcascade_frontface.xml')

while(True):
    ret, cam = cap.read()
    frame = cv2.flip(cam, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.05, 5)

    if len(faces):
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            image_blur = cv2.blur(face,(50,50))
            frame[y:y+h, x:x+w] = image_blur
            cv2.imshow('camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
