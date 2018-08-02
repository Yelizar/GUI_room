import cv2
import numpy as np
import os


class ImgReader():
    """return processed pictures"""
    def __init__(self):

        self.img_pack = self.find_img()
        self.point = 0
        self.crt_image = None
        self.movement_point(None)


    def find_img(self):
        """create a list of images and return it"""
        #find right path to the folder with images
        profect_dir = os.path.curdir
        data = os.path.join(profect_dir, 'data')
        glasses_dir = os.path.join(data, 'glasses')
        #format of path ---> current directory \ glasses folder \ image name
        pack = [glasses_dir+'\\'+img for img in os.listdir(glasses_dir) \
                         if os.path.isfile(os.path.join(glasses_dir, img))]
        return pack

    def movement_point(self, diff, *args):
        """This method changes the index of the image being processed"""
        if diff:
            self.point += diff
            if len(self.img_pack) <= self.point:
                self.point = 0
            elif self.point < 0:
                self.point = (len(self.img_pack) - 1)
            self.crt_image = self.img_pack[self.point]
        else:
            self.crt_image = self.img_pack[self.point]

    def png_reader(self, image):
        """This method is processing png image with alpha channel and return it in correct format"""
        rows, coloms, channels = image.shape
        if channels == 4:
            """RGBA"""
            # split image on two alpha and rgb channel
            alpha_cnl = image[:,:,3]
            rgb_cnl = image[:,:,:3]
            #White Background Image
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
            #create alpha channel for RGB/BGR image. once Alpha channel created, use it for alpha factor.
            alpha_cnl = np.ones(image.shape, dtype=np.uint8) * 255
            base = image.astype(np.float32) / alpha_cnl
            return base.astype(np.uint8)

    def cvt_gray(self, image):
        """This method converts image from RGB to GRAY"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray

    def img_read(self, rqs_img, *args):
        """This method gets an image's path and returns processed picture"""
        if rqs_img:
            image = cv2.imread(str(rqs_img), cv2.IMREAD_UNCHANGED)
        else:
            image = cv2.imread(str(self.img_pack[0]), cv2.IMREAD_UNCHANGED)
            self.crt_image = self.img_pack.index('.\\data\\glasses\\glasses_1.png')
        image = self.png_reader(image)
        return image

    def right_btn(self,  *args):
        """Switch images -->"""
        self.movement_point(1)

    def left_btn(self, *args):
        """Switch images <--"""
        self.movement_point(-1)
