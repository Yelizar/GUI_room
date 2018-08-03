import cv2
import numpy as np

from moduls import resize_image, cvt_gray, masks_of_image


def processing(frame, face, image):
    for (x, y, w, h) in face:
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
    return frame
