#made_by_Lazarus_Rai
#Version 0.1
import kivy
kivy.require('1.10.0')

from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button

from img_reader import ImgReader
import cv2
import os, sys


class Screen(Image, BoxLayout):
    """Outputs screen"""
    def __init__(self, capture, fps, **kwargs):
        super(Screen, self).__init__(**kwargs)
        self.img_reader = ImgReader()
        # ^Button PREVIOUS
        self.previous = Button(text='Previous',
                               pos=(0, 260),
                               size=(80, 80),
                               size_hint=(None, None),
                               background_color=(10, 10, 10, 0.1)
                               )
        self.previous.bind(on_press=self.img_reader.left_btn)
        Screen.add_widget(self, self.previous)
        # $
        # ^Button NEXT
        self.next = Button(text='Next',
                           pos=(0, 0),
                           size=(80, 80),
                           size_hint=(None, None),
                           background_color=(10, 10, 10, 0.1)
                           )
        self.next.bind(on_press=self.img_reader.right_btn)
        Screen.add_widget(self, self.next)
        # $

        self.crt_image = self.img_reader.crt_image
        self.prc_crt_image = None
        self.capture = capture
        self.face_cascade = cv2.CascadeClassifier('data/face.xml')
        self.eyes_cascade = cv2.CascadeClassifier('data/eyes.xml')

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
            face = self.face_cascade.detectMultiScale(frame_gray, 1.3, 5)
            for (x, y, w, h) in face:
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                #cut face from the frame for further work
                cuted_face = frame[y+75:y+(h//2),x+20:x+w-20]
                cuted_face_gray = self.img_reader.cvt_gray(cuted_face)
                eyes = self.eyes_cascade.detectMultiScale(cuted_face_gray, 1.3, 5)
                if len(eyes) == 2:
                    for (xe, ye, we, he) in eyes:
                        cv2.rectangle(cuted_face, (xe, ye), (xe+we, ye+he), (233,133,233), 2)

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


def resourcePath():
    '''Returns path containing content - either locally or in pyinstaller tmp file'''
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS)

    return os.path.join(os.path.abspath("."))


if __name__ == '__main__':
    kivy.resources.resource_add_path(resourcePath())
    my_app = Main().run()