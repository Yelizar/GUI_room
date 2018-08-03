import cv2
import numpy as np


class Cascade():
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('data/face.xml')
        self.eyes_cascade = cv2.CascadeClassifier('data/eyes.xml')

    def face(self, frame_gray):
        face = self.face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        return face



        # eyes = self.eyes_cascade.detectMultiScale(cuted_face_gray, 1.3, 5)
        # if len(eyes) == 2:
        #     for (xe, ye, we, he) in eyes:
        #         cv2.rectangle(cuted_face, (xe, ye), (xe + we, ye + he), (233, 133, 233), 2)