import cv2
import numpy as np

from moduls import find_img, png_reader


class ImgReader:
    """return processed pictures"""
    def __init__(self):
        self.img_pack = find_img()
        self.point = 0
        self.crt_image = None
        self.movement_point()

    def movement_point(self, diff=None, *args):
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

    def img_read(self, rqs_img, *args):
        """This method gets an image's path and returns processed picture"""
        if rqs_img:
            image = cv2.imread(str(rqs_img), cv2.IMREAD_UNCHANGED)
        else:
            image = cv2.imread(str(self.img_pack[0]), cv2.IMREAD_UNCHANGED)
            self.crt_image = self.img_pack.index('./data/glasses/glasses_1.png')
        image = png_reader(image)
        return image

    def right_btn(self,  *args):
        """Switch images -->"""
        self.movement_point(1)

    def left_btn(self, *args):
        """Switch images <--"""
        self.movement_point(-1)
