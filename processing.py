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


def imcrop(img, x1, y1, x2, y2):

    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
                       (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0, 0)), mode="constant")
    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    return img, x1, x2, y1, y2

class PostPrc:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('data/face.xml')
        self.eyes_cascade = cv2.CascadeClassifier('data/eyes.xml')
        self.angle = 45
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

    def rotation(self, x, y, w, h):

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
        rows, cols, _ = self.image.shape
        rot = cv2.getRotationMatrix2D((((rows / 2),(cols / 2))), self.angle, 1)
        dst = cv2.warpAffine(self.image, M, (cols, rows), flags=cv2.INTER_LINEAR)
        print(dst)
        return dst

    def processing(self, frame, image):
        """The function processed the data(Frame, the cascade and an image) and has to return a frame"""
        frame_gray = cvt_gray(frame)
        self.crt_position_cascade = np.array(self.cut_obj(frame_gray))
        self.cascade_verification()
        for (x, y, w, h) in self.crt_position_cascade:
            # cut face from the frame for further work
            cuted_face = frame[y:y + h, x:x + w]
            # eyes = np.array((self.cut_obj(cuted_face)))
            # for (x_eye, y_eye, w_eye, h_eye) in eyes:
            #     print(x_eye, y_eye, w_eye, h_eye)
            # resize image in accordance with the size of the face

            self.image = resize_image(image, w, h)
            self.image = self.rotation(x, y, w, h)
            # create mask and inversion mask of the image
            image_gray = cvt_gray(self.image)
            mask, mask_inv = masks_of_image(image_gray)
            print(self.image.shape)

            # add image to frame
            frame_bg = cv2.bitwise_and(cuted_face, cuted_face, mask=mask)
            image_fg = cv2.bitwise_and(self.image, self.image, mask=mask_inv)
            self.image = cv2.add(frame_bg, image_fg)
            frame[y:y + h, x:x + w] = self.image
            self.prv_position_cascade = self.crt_position_cascade

        return frame
