# -*- coding: utf-8 -*-
import cv2
# import face_recognition
import os
#practice

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


face_detector = cv2.CascadeClassifier('haarcascade_frontface.xml')

def blurBoxes(image, boxes):

    for (x,y,w,h) in boxes:
        # crop the image due to the current box
        sub = image[y:y+h, x:x+w]

        # apply GaussianBlur on cropped area
        # blur = cv2.GaussianBlur(sub,(3,3),0)
        blur = cv2.blur(sub, (25,25))

        # paste blurred image on the original image
        image[y:y+h, x:x+w] = blur

    return image

while(True):
    ret, cam = cap.read()
    if cam is None:
        print('Camera load failed')
        break

    frame = cv2.flip(cam, 1)
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector.detectMultiScale(RGB)

    frame_blur = blurBoxes(frame, faces)
    # print("Num of faces: ", str(len(faces)))

    if len(faces):
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w, y+h),(255,0,0),2)

    if(ret) :
        cv2.imshow('camera', frame_blur)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


                
cap.release()
cv2.destroyAllWindows()
