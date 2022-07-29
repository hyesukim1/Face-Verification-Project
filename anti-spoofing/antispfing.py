# -*- coding: utf-8 -*-
from keras.models import load_model
import cv2
import numpy as np

model = load_model('C:/Users/USER/OneDrive - 주식회사 인티그리트/문서/GitHub/Face-Verification-Project/anti-spoofing/antispoofing.h5')
print(model.summary())

face_detector = cv2.CascadeClassifier('haarcascade_frontface.xml')

cap = cv2.VideoCapture(0) # connect usb camera

def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)

while True:
    ret, frame = cap.read()
    if ret is False:
        print "Error happens from camera"

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        
        roi = frame[y:y+h, x:x+w]

        img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
        img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)

        ycrcb_hist = calc_hist(img_ycrcb)
        luv_hist = calc_hist(img_luv)

        
        cv2.imshow('camera', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()       