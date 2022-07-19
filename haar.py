import cv2
import face_recognition
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

face_detector = cv2.CascadeClassifier('C:/Users/USER/face_reco/haarcascade_frontface.xml')

while(True):
    ret, cam = cap.read()
    frame = cv2.flip(cam, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.05, 5)
    print("Num of faces: ", str(len(faces)))

    if len(faces):
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w, y+h),(255,0,0),2)

    if(ret) :
        cv2.imshow('camera', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                     
cap.release()
cv2.destroyAllWindows()
