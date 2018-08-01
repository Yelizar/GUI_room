#made_by_Lazarus_Rai
import kivy
kivy.require('1.10.0')

from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.lang import Builder

from itertools import cycle, tee, islice, chain, zip_longest

import cv2
import numpy as np
import os, sys


class ImgReader(BoxLayout):
    """return processed pictures"""
    def __init__(self, **kwargs):
        super(ImgReader, self).__init__(**kwargs)
        self.img_pack = self.find_img()
        self.point = 0
        self.crt_image = None
        self.movement_point(None)

        # ^Button PREVIOUS
        self.previous = Button(text='Previous',
                               pos=(0, 260),
                               size=(80, 80),
                               size_hint=(None, None),
                               background_color=(10, 10, 10, 0.1)
                               )
        self.previous.bind(on_press=self.left_btn)
        ImgReader.add_widget(self, self.previous)
        # $
        # ^Button NEXT
        self.next = Button(text='Next',
                           pos=(0, 0),
                           size=(80, 80),
                           size_hint=(None, None),
                           background_color=(10, 10, 10, 0.1)
                           )
        self.next.bind(on_press=self.right_btn)
        ImgReader.add_widget(self, self.next)
        # $

    def find_img(self):
        """create a list of images and return it"""
        #find right path to the folder with images
        profect_dir = os.path.curdir
        glasses_dir = os.path.join(profect_dir, 'glasses')
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
            # split image on two alpha and rgb channel
            alpha_cnl = image[:,:,3]
            rgb_cnl = image[:,:,:3]
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
            b_channel, g_channel, r_channel = cv2.split(image)
            white_background_image = np.ones_like(image, dtype=np.uint8) * 255
            #create alpha channel for RGB/BGR image. once Alpha channel created, use it for alpha factor.
            alpha_cnl = np.ones(b_channel.shape, dtype=np.uint8) * 1  # creating a dummy alpha channel image.
            alpha_factor = alpha_cnl[:,:,np.newaxis].astype(np.float32) * 255

            alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)
            base = image.astype(np.float32) - alpha_factor
            white = white_background_image * (1 - alpha_factor)
            white = image * (1 - white)
            final_image = base + white

            return alpha_factor.astype(np.uint8)

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
            self.crt_image = self.img_pack.index('.\\glasses\\glasses_1.png')
        image = self.png_reader(image)
        return image

    def right_btn(self,  *args):
        """Switch images -->"""
        self.movement_point(1)

    def left_btn(self, *args):
        """Switch images <--"""
        self.movement_point(-1)



class Screen(Image):
    """Outputs screen"""
    def __init__(self, capture, fps, **kwargs):
        super(Screen, self).__init__(**kwargs)
        self.img_reader = ImgReader()
        self.crt_image = self.img_reader.crt_image
        self.prc_crt_image = None
        self.capture = capture
        self.face_cascade = cv2.CascadeClassifier('face.xml')
        self.add_widget(self.img_reader)
        Clock.schedule_interval(self.update, 1.0 / fps)

    def img_verification(self):
        """This method checks which image should be processed"""
        if self.crt_image is None:
            self.prc_crt_image = self.img_reader.img_read(None)
            self.crt_image = self.img_reader.crt_image
        elif self.crt_image is self.img_reader.crt_image:
            if self.prc_crt_image is not None:
                pass
            else:
                self.prc_crt_image = self.img_reader.img_read(None)
                self.crt_image = self.img_reader.crt_image
        else:
            self.crt_image = self.img_reader.crt_image
            self.prc_crt_image = self.img_reader.img_read(self.crt_image)
        return self.prc_crt_image

    def resize_image(self, image, width, height):
        """This method returns modified image with the size of the face """
        modified_image = cv2.resize(image, dsize=(width, height))
        return modified_image

    def masks_of_image(self, image_gray):
        """This method convert image from RGB to GRAY and return mask and inversion of the image"""
        ret, mask = cv2.threshold(image_gray, 0, 254, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        return mask, mask_inv

    def update(self, dt):
        """processing video frames"""
        ret, frame = self.capture.read()
        if ret:
            # convert bgr frame into gray
            frame_gray = self.img_reader.cvt_gray(frame)
            # detect face in gray frame
            self.face = self.face_cascade.detectMultiScale(frame_gray, 1.3, 5)

            for (x, y, w, h) in self.face:
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                #cut face from the frame for further work
                cuted_face = frame[y+75:y+(h//2),x+20:x+w-20]
                image = self.img_verification()
                #resize image in accordance with the size of the face
                image = self.resize_image(image, w-40, h//2-75)

                #create mask and inversion mask of the image
                image_gray = self.img_reader.cvt_gray(image)
                mask, mask_inv = self.masks_of_image(image_gray)

                #add image to frame
                frame_bg = cv2.bitwise_and(cuted_face, cuted_face, mask=mask)
                image_fg = cv2.bitwise_and(image, image, mask=mask_inv)
                image = cv2.add(frame_bg, image_fg)
                frame[y+75:y+(h//2), x+20:x + w-20] = image

            # convert it to texture
            buf1 = cv2.flip(frame, -1)
            buf = buf1.tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture


class Main(App, BoxLayout):


    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.my_camera = Screen(capture=self.capture, fps=30)
        return self.my_camera


    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.capture.release()


if __name__ == '__main__':
    Main().run()