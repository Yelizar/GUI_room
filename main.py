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
from cascade import Cascade
from processing import Processing

import cv2
import os, sys


class Screen(Image, BoxLayout):
    """Outputs screen"""
    def __init__(self, capture, fps, **kwargs):
        super(Screen, self).__init__(**kwargs)
        self.img_reader = ImgReader()
        self.cascade = Cascade()
        self.processing = Processing()
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



    def update(self, dt):
        """processing video frames"""
        ret, frame = self.capture.read()
        image = self.img_verification()
        if ret:
            # convert bgr frame into gray
            frame_gray = self.img_reader.cvt_gray(frame)
            # detect face in gray frame
            face = self.cascade.face(frame_gray)
            # processing - return final image
            frame = self.processing.processing(frame, face, image)
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