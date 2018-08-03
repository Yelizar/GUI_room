import cv2
import numpy as np

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
        self.eyes_cascade = cv2.CascadeClassifier('data/eyes.xml')
        self.crt_position_cascade = None
        self.prv_position_cascade = None

    def cascade_verification(self):
        if self.prv_position_cascade is not None and \
                self.prv_position_cascade.shape == self.crt_position_cascade.shape:
            diff = self.crt_position_cascade - self.prv_position_cascade
            if (abs(max(diff[0]))) < 4:
                self.crt_position_cascade = self.prv_position_cascade

    def face(self, frame_gray):
        face = self.face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        return face

    def processing(self, frame, image):
        """The function processed the data(Frame, the cascade and an image) and has to return a frame"""
        frame_gray = cvt_gray(frame)
        self.crt_position_cascade = np.array(self.face(frame_gray))
        self.cascade_verification()
        for (x, y, w, h) in self.crt_position_cascade:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            # cut face from the frame for further work
            cuted_face = frame[y + 75:y + (h // 2), x + 20:x + w - 20]

            image = image
            # resize image in accordance with the size of the face
            image = resize_image(image, w - 40, h // 2 - 75)

            # create mask and inversion mask of the image
            image_gray = cvt_gray(image)
            mask, mask_inv = masks_of_image(image_gray)

            # add image to frame
            frame_bg = cv2.bitwise_and(cuted_face, cuted_face, mask=mask)
            image_fg = cv2.bitwise_and(image, image, mask=mask_inv)
            image = cv2.add(frame_bg, image_fg)
            frame[y + 75:y + (h // 2), x + 20:x + w - 20] = image
            self.prv_position_cascade = self.crt_position_cascade

        return frame
