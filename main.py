#made_by_Lazarus_Rai
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button

import cv2
import numpy as np
import os, sys

def cvt_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


class ImgReader(BoxLayout):
    """return processed pictures"""
    def __init__(self, **kwargs):
        super(ImgReader, self).__init__(**kwargs)
        self.img_pack = self.find_img()
        print(self.img_pack[0])

    def find_img(self):
        """create a list of images and return it"""
        #find right path to the folder with images
        profect_dir = os.path.curdir
        glasses_dir = os.path.join(profect_dir, 'glasses')
        #format of path ---> current directory \ glasses folder \ image name
        img_pack = [glasses_dir+'\\'+img for img in os.listdir(glasses_dir) \
                         if os.path.isfile(os.path.join(glasses_dir, img))]
        return img_pack

    def png_reader(self, image):
        """This method is processing png image with alpha channel and return it in correct format"""
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

    def img_read(self):
        image = cv2.imread(str(self.img_pack[0]), cv2.IMREAD_UNCHANGED)
        glasses = self.png_reader(image)
        return glasses


    def right_btn(self):
        self.cur_img = self.cur_img + 1
        self.img_read()



class Camera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(Camera, self).__init__(**kwargs)
        self.image = ImgReader()
        self.capture = capture
        self.face_cascade = cv2.CascadeClassifier('face.xml')
        Clock.schedule_interval(self.update, 1.0 / fps)

    def resize_image(self, image, width, height):
        """This method returns modified image with the size of the face """
        modified_image = cv2.resize(image, dsize=(width, height))
        return modified_image

    def masks_of_image(self, image):
        """This method convert image from RGB to GRAY and return mask and inversion of the image"""
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, mask = cv2.threshold(image_gray, 0, 254, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        return mask, mask_inv

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
                cuted_face = frame[y:y+h,x:x+w]
                #get image
                image = self.image.img_read()
                #resize image in accordance with the size of the face
                image = self.resize_image(image, w, h)

                #create mask and inversion mask of the image
                mask, mask_inv = self.masks_of_image(image)

                #add image to frame
                frame_bg = cv2.bitwise_and(cuted_face, cuted_face, mask=mask)
                image_fg = cv2.bitwise_and(image, image, mask=mask_inv)
                image = cv2.add(frame_bg, image_fg)
                frame[y:y + h, x:x + w] = image

            # convert it to texture
            buf1 = cv2.flip(frame, -1)
            buf = buf1.tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
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