import cv2
import numpy as np

class Processing():
    def __init__(self):
        pass



    def resize_image(self, image, width, height):
        """This method returns modified image with the size of the face """
        modified_image = cv2.resize(image, dsize=(width, height))
        return modified_image

    def masks_of_image(self, image_gray):
        """This method convert image from RGB to GRAY and return mask and inversion of the image"""
        ret, mask = cv2.threshold(image_gray, 0, 254, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        return mask, mask_inv

    def cvt_gray(self, image):
        """This method converts image from RGB to GRAY"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray

    def processing(self, frame, face, image):
        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            # cut face from the frame for further work
            cuted_face = frame[y + 75:y + (h // 2), x + 20:x + w - 20]

            image = image
            # resize image in accordance with the size of the face
            image = self.resize_image(image, w - 40, h // 2 - 75)

            # create mask and inversion mask of the image
            image_gray = self.cvt_gray(image)
            mask, mask_inv = self.masks_of_image(image_gray)

            # add image to frame
            frame_bg = cv2.bitwise_and(cuted_face, cuted_face, mask=mask)
            image_fg = cv2.bitwise_and(image, image, mask=mask_inv)
            image = cv2.add(frame_bg, image_fg)
            frame[y + 75:y + (h // 2), x + 20:x + w - 20] = image
        return frame
