
cap = cv2.VideoCapture(0)


while(True):
    ret, cam = cap.read()

    
    cv2.imshow('camera', cam)
