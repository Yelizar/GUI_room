#made_by_Lazarus_Rai
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import cv2
import os, sys

def cvt_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


class ImgReader():
    """return processed pictures"""
    def __init__(self):
        self.img_pack = []

    def find_img(self):
        """creat a list of images"""
        #find right oath to the folder with images
        profect_dir = os.path.expanduser('~')
        glasses_dir = os.path.join(profect_dir, 'glasses')



class Camera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(Camera, self).__init__(**kwargs)
        self.capture = capture
        self.face_cascade = cv2.CascadeClassifier('face.xml')
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        """processing video frames"""
        ret, frame = self.capture.read()
        if ret:
            # convert bgr frame into gray
            frame_gray = cvt_gray(frame)
            # detect face in gray frame
            self.face = self.face_cascade.detectMultiScale(frame_gray, 1.3, 5)

            for (x, y, w, h) in self.face:
                cv2.rectangle(frame, (x, y), (x + w, y + (h // 2)), (255, 255, 0), 2)
                #cut face from the frame for further work
                crd_face = frame[y:y+h,x:x+w]
                cuted_face = crd_face
                cuted_gray_face = frame_gray[y:y+h,x:x+w]


            # convert it to texture
            buf1 = cv2.flip(frame, -1)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture



class Main(App):


    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.my_camera = Camera(capture=self.capture, fps=30)
        return self.my_camera

    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.capture.release()


if __name__ == '__main__':
    Main().run()