import cv2
import numpy as np
import math


from moduls import cvt_gray


def resize_image(image, width, height):
    """This function returns modified image with the size of the face """
    modified_image = cv2.resize(image, dsize=(width, height))
    return modified_image


def masks_of_image(image_gray):
    """This function convert image from RGB to GRAY and return mask and inversion of the image"""
    ret, mask = cv2.threshold(image_gray, 0, 254, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    return mask, mask_inv


class PostPrc:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('data/face.xml')
        self.eyes_cascade = cv2.CascadeClassifier('data/left_eye.xml')
        self.angle = 0
        self.image = None
        self.crt_position_cascade = None
        self.prv_position_cascade = None

    def cascade_verification(self):
        if self.prv_position_cascade is not None and \
                self.prv_position_cascade.shape == self.crt_position_cascade.shape:
            diff = self.crt_position_cascade - self.prv_position_cascade
            if (abs(max(diff[0]))) == 1:
                self.crt_position_cascade = self.prv_position_cascade

    def cut_obj(self, frame_gray):
        obj = self.face_cascade.detectMultiScale(frame_gray, scaleFactor=1.02, minNeighbors=10, minSize=(150, 150))
        return obj

    def rotation(self):
        """
        self.x_1 = x * (math.cos(self.angle)) - y * (math.sin(self.angle))
        self.y_1 = x * (math.cos(self.angle)) + y * (math.sin(self.angle))
        self.x_2 = (x + w) * (math.cos(self.angle)) - y * (math.sin(self.angle))
        self.y_2 = (x + w) * (math.cos(self.angle)) + y * (math.sin(self.angle))
        self.x_3 = x * (math.cos(self.angle)) - (y + h) * (math.sin(self.angle))
        self.y_3 = x * (math.cos(self.angle)) + (y + h) * (math.sin(self.angle))
        self.x_4 = (x + w) * (math.cos(self.angle)) - (y + h) * (math.sin(self.angle))
        self.y_4 = (x + w) * (math.cos(self.angle)) + (y + h) * (math.sin(self.angle))
        pts1 = np.float32([[x, y], [x + w, y], [x, y + h]])
        pts2 = np.float32([[self.x_1, self.y_1], [self.x_2, self.y_2], [self.x_3, self.y_3]])
        M = cv2.getAffineTransform(pts1, pts2)
        """

        rows, cols, _ = self.image.shape
        rot = cv2.getRotationMatrix2D((((rows / 2),(cols / 2))), ((1 - self.angle) * 3), 1)
        dst = cv2.warpAffine(self.image, rot, (cols, rows), flags=cv2.INTER_LINEAR)
        return dst

    def processing(self, frame, image):
        """The function processed the data(Frame, the cascade and an image) and has to return a frame"""
        frame_gray = cvt_gray(frame)
        self.crt_position_cascade = np.array(self.cut_obj(frame_gray))

        self.cascade_verification()
        for (x, y, w, h) in self.crt_position_cascade:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cut face from the frame for further work
            cuted_face = frame[y:y + h, x:x + w]
            # resize image in accordance with the size of the faceo**
            eyes = self.eyes_cascade.detectMultiScale(cuted_face, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 233, 233), 2)
                angle = (ey-y) / math.hypot((ey-y), (ex-x))
                self.angle = math.degrees(angle) - 20
                print(self.angle)

            else:
                pass
            self.image = resize_image(image, w, h)
            self.image = self.rotation()
            # create mask and inversion mask of the image
            image_gray = cvt_gray(self.image)
            mask, mask_inv = masks_of_image(image_gray)
            # add image to frame
            frame_bg = cv2.bitwise_and(cuted_face, cuted_face, mask=mask)
            image_fg = cv2.bitwise_and(self.image, self.image, mask=mask_inv)
            self.image = cv2.add(frame_bg, image_fg)

            frame[y:y + h, x:x + w] = self.image

            self.prv_position_cascade = self.crt_position_cascade

        return frame