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
        self.eyes_cascade = cv2.CascadeClassifier('data/eyes.xml')
        self.angle = 15
        self.image = None
        self.crt_position_cascade = None
        self.prv_position_cascade = None

    def cascade_verification(self):
        if self.prv_position_cascade is not None and \
                self.prv_position_cascade.shape == self.crt_position_cascade.shape:
            diff = self.crt_position_cascade - self.prv_position_cascade
            if (abs(max(diff[0]))) == 1:
                self.crt_position_cascade = self.prv_position_cascade

    def cut_obj(self, frame_gray, factor, _min):
        obj = self.face_cascade.detectMultiScale(frame_gray, scaleFactor=factor, minNeighbors=10, minSize=(_min, _min))
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
        rot = cv2.getRotationMatrix2D((((rows / 2),(cols / 2))), self.angle, 1)
        dst = cv2.warpAffine(self.image, rot, (cols, rows), flags=cv2.INTER_LINEAR)
        return dst

    def processing(self, frame, image):
        """The function processed the data(Frame, the cascade and an image) and has to return a frame"""
        frame_gray = cvt_gray(frame)
        self.crt_position_cascade = np.array(self.cut_obj(frame_gray, 1.02, 150))
        eyes = self.eyes_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=20, minSize=(30, 30),
                                                                                                    maxSize=(75, 75))
        self.cascade_verification()
        for (x, y, w, h) in self.crt_position_cascade:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cut face from the frame for further work
            cuted_face = frame[y:y + h, x:x + w]

            if eyes:
                print(1)
                ex, ey, ew, eh = eyes[0]
                ex_1, ey_1, ew_1, eh_1 = eyes[1]
                print(2)
            # resize image in accordance with the size of the face

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

    # def processing(self, frame, image):
    #     """The function processed the data(Frame, the cascade and an image) and has to return a frame"""
    #     eyes = self.eyes_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=20, minSize=(30, 30),
    #                                               maxSize=(75, 75))
    #     counter = 0
    #     for eye in eyes:
    #         ex, ey, ew, eh = eye
    #         cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    #         # base = frame[ey:ey + eh, ex:ex + ew]
    #         # self.image = resize_image(image, ew, eh)
    #         # self.image = self.rotation()
    #         # # create mask and inversion mask of the image
    #         # image_gray = cvt_gray(self.image)
    #         # mask, mask_inv = masks_of_image(image_gray)
    #         # # add image to frame
    #         # frame_bg = cv2.bitwise_and(base, base, mask=mask)
    #         # image_fg = cv2.bitwise_and(self.image, self.image, mask=mask_inv)
    #         # self.image = cv2.add(frame_bg, image_fg)
    #         #
    #         # frame[ey:ey + eh, ex:ex + ew] = self.image
    #         #
    #         # self.prv_position_cascade = self.crt_position_cascade
    #
    #         if counter == 0 and len(eyes) == 2:
    #             counter += 1
    #             ex_1, ey_1, ew_1, eh_1 = ex, ey, ew, eh
    #         elif counter == 1 and len(eyes) == 2: counter = 0
    #
    #     return frame
