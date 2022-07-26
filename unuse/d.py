# -*- coding: utf-8 -*-
import dlib
import cv2
import numpy as np

# num of face landmark
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))

# face detector, predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(1)
# 회사 웹캠은 1, 노트북 카메라는 0

while True:

    ret, cam = cap.read()

    image = cv2.resize(cam, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_detector = detector(img_gray, 1)
    # the number of face detected
    print("The number of faces detected : {}".format(len(face_detector)))

    for face in face_detector:
        # face wrapped with rectangle
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()),(0, 0, 255), 3)

        landmarks = predictor(image, face)  # find 68 landmark from face

        landmark_list = []
        # append (x, y) in landmark_list
        for p in landmarks.parts():
            landmark_list.append([p.x, p.y])
            cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), -1)


    cv2.imshow('result', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

with open("test.json", "w") as json_file:
	key_val = [ALL, landmark_list]
	landmark_dict = dict(zip(*key_val))
	print(landmark_dict)
	json_file.write(json.dumps(landmark_dict))
	json_file.write('\n')

cap.release()

'''
거리 리스트를 담고
for a in 거리리스트
if 리스트=실제거리
비디오 continue
or
얼굴 bbox 크기가 일정부분 작아지면 인지
'''