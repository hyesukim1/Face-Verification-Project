import cv2

cap = cv2.VideoCapture(1)

while(True):
    ret, cam = cap.read()
    if cam is None:
        print('Camera load failed')
        break


    cv2.imshow('camera', cam)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()