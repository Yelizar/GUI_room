import os
import numpy as np
import cv2


def find_img():
    """create a list of images and return it"""
    # find right path to the folder with images
    profect_dir = os.path.curdir
    data = os.path.join(profect_dir, 'data')
    glasses_dir = os.path.join(data, 'glasses')
    # format of path ---> current directory \ glasses folder \ image name
    pack = [glasses_dir + '\\' + img for img in os.listdir(glasses_dir) \
            if os.path.isfile(os.path.join(glasses_dir, img))]
    return pack


def png_reader(image):
    """This function is processing png image with alpha channel and return it in correct format"""
    rows, coloms, channels = image.shape
    if channels == 4:
        """RGBA"""
        # split image on two alpha and rgb channel
        alpha_cnl = image[:, :, 3]
        rgb_cnl = image[:, :, :3]
        # White Background Image
        white_background_image = np.ones_like(rgb_cnl, dtype=np.uint8) * 255
        # Alpha factor
        alpha_factor = alpha_cnl[:, :, np.newaxis].astype(np.float32) / 255
        alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)
        # Transparent Image Rendered on White Background
        base = rgb_cnl.astype(np.float32) + (1 - alpha_factor)
        white = white_background_image.astype(np.float32) * (1 - alpha_factor)
        white = rgb_cnl * (1 - white)
        final_image = base + white
        return final_image.astype(np.uint8)
    else:
        """RGB with white background"""
        # Create alpha channel for RGB/BGR image. once Alpha channel created, use it for alpha factor.
        alpha_cnl = np.ones(image.shape, dtype=np.uint8) * 255
        base = image.astype(np.float32) / alpha_cnl
        return base.astype(np.uint8)


def cvt_gray(image):
    """This function converts image from RGB to GRAY"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray


def resize_image(image, width, height):
    """This function returns modified image with the size of the face """
    modified_image = cv2.resize(image, dsize=(width, height))
    return modified_image


def masks_of_image(image_gray):
    """This function convert image from RGB to GRAY and return mask and inversion of the image"""
    ret, mask = cv2.threshold(image_gray, 0, 254, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    return mask, mask_inv
