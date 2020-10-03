from vidmodels import return_age, return_gender
import cv2, time
import numpy as np

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    _, frame = capture.read()
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gframe, 1.3,5)
    for (x,y,w,h) in faces:
        fc = gframe[y:y+h, x:x+w]
        roi = cv2.resize(fc, (200,200))
        rgbroi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        p1 = return_age.return_age(rgbroi)
        rgbroi = cv2.resize(rgbroi, (261,195))
        p2 = return_gender.return_gender(rgbroi)
        text = str(p1) + " / " + str(p2)
        cv2.putText(frame, text, (x,y),font, 1, (255,0,0), 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.imshow('Video', frame)
        cv2.waitKey(1000)
