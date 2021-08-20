import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import morphology
from PIL import Image





class OwnImageTester:
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrtuvwxyz'
    img = np.array([])
    height = 28
    width = 28
    
    def __init__(self, path_to_img):
        #load grayscale image
        self.img = Image.open(path_to_img).convert('L')
        self.img = np.array(self.img)
        
    def normalize(self):
        self.resize()
        self.convert_to_bin()
        self.detect_background()
        self.delete_remains()
        self.center_character()
        
    def resize(self):
        #resize to 28 x 28 (5 iterations because it isn't missing accuracy of image)
        resizes = (280, 210, 140, 70, 28)
        for i in resizes:
            self.img = cv2.resize(self.img, (i, i))

    def convert_to_bin(self):
        #get mean that be able to separate differents colors in image
        mean = np.mean(self.img)
        #change each pixel to 0 or 255. Dependent on mean: 
        #if less than mean - 0
        #if greater than mean - 255
        _, self.img = cv2.threshold(self.img, mean, 255, cv2.THRESH_BINARY)
        
    def detect_background(self):
        #detect color of background
        pixels0 = np.count_nonzero(self.img == 0)
        pixels255 = np.count_nonzero(self.img == 255)
        #those pixels with greater quantity are arguably a background
        if pixels255 > pixels0:
            #swap for getting real background 
            #it can be swapped (depent on input colors)
            self.img = np.abs(self.img - 255)

    def delete_remains(self):
        #delete small objects if they exist (especially single pixels)
        self.img = self.img / 255
        self.img = self.img.astype('bool')
        self.img = morphology.remove_small_objects(self.img, min_size=3)
        
    def center_character(self):
        #last step is center characters with removing empty lines and fill them again 
        #(both horizontal and vertical)
        
        #removing
        mask = self.img == 0
        rows = np.flatnonzero((~mask).sum(axis=1))
        cols = np.flatnonzero((~mask).sum(axis=0))
        self.img = self.img[rows.min():rows.max()+1, cols.min():cols.max()+1]
        
        #filling
        shape = self.img.shape
        #count numbers of lines to add in height and width
        v_additional_lines_num = int((self.height - shape[0]) / 2)
        h_additional_lines_num = int((self.width - shape[1]) / 2)
        #preparing row
        line = np.zeros(shape[1])
        #adding rows
        for i in range(v_additional_lines_num):
            self.img = np.vstack([line, self.img])
            self.img = np.vstack([self.img, line])
        #if dim has been odd num, it's necessery to add 1 more
        if self.img.shape[0] == 27:
            self.img = np.vstack([self.img, line])
        #as above but in horizontal
        line = np.zeros([self.height, 1])
        for i in range(h_additional_lines_num):
            self.img = np.hstack([line, self.img])
            self.img = np.hstack([self.img, line])
        if self.img.shape[1] == 27:
            self.img = np.hstack([self.img, line])        
        
    def test(self, model, with_plot=True):
        if with_plot:
            #show converted image
            plt.imshow(self.img)
            plt.show()
        #predict
        self.img = self.img.reshape(1, self.height, self.width, 1)
        res = model.predict(self.img)
        num = np.argmax(res[0])
        return self.characters[num]



