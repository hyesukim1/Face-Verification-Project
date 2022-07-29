# -*- coding: utf-8 -*-
import cv2
import numpy as np

def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)

image = 'C:/Users/USER/OneDrive - 주식회사 인티그리트/문서/GitHub/Face-Verification-Project/anti-spoofing/11.jpg'
hist = calc_hist(image)
print(hist)