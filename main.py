import matplotlib.pyplot as plt
from emnist import extract_training_samples, extract_test_samples
from tensorflow.keras import Sequential, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow import nn
import cv2
import numpy as np
from skimage import morphology
from PIL import Image












class HandwrittenCharactersRecognizer:
    #number of classes (actualy length of string containing all letters and digits)
    classes_number = 61
    #name of dataset to use
    dataset_name = 'balanced'
    #dimensions of single sample
    height = 28
    width = 28
    
    def __init__(self):
        #train data
        self.x_train = list()
        self.y_train = list()
        #test data
        self.x_test = list()
        self.y_test = list()
        #main model
        self.model = Sequential()
        #history of fit
        self.history = None
    
    def load_data(self):
        #load train and test data from emnist package
        self.x_train, self.y_train = extract_training_samples(self.dataset_name)
        self.x_test, self.y_test = extract_test_samples(self.dataset_name)
        
    def normalize_data(self):
        #this function is actualy resizing x data
        #and change y data to vectors with only once 1 number in appropriate position
        #train
        self.x_train = self.x_train / 255.0
        samples_quantity = self.x_train.shape[0]
        self.x_train = self.x_train.reshape(samples_quantity, self.height, self.width, 1)
        self.y_train = to_categorical(self.y_train, self.classes_number)
        #test
        samples_quantity = self.x_test.shape[0]
        self.x_test = self.x_test.reshape(samples_quantity, self.height, self.width, 1)
        self.y_test = to_categorical(self.y_test, self.classes_number)
        
    def build_model(self, show=False):
        #building a model
        input_shape = (self.height, self.width, 1)
        self.model.add(layers.Flatten(input_shape=input_shape))
        self.model.add(layers.Dense(512, activation=nn.relu))
        self.model.add(layers.Dense(256, activation=nn.relu))
        self.model.add(layers.Dense(128, activation=nn.relu))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(self.classes_number, activation=nn.softmax))
        if show:
            self.model.summary()
            
    def fit(self, save=False):
        self.model.compile(
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        self.history = self.model.fit(
            self.x_train, 
            self.y_train, 
            epochs=6, 
            batch_size=256
        )
        if save:
            self.model.save('model.h5')
        
    def test(self):
        outcome = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("Accuracy: %.2f%%"%(outcome[1] * 100))
        accuracy = outcome[1] * 100
        return accuracy
    
    def load_model(self):
        self.model = load_model('model.h5')
        









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
    
    
    
    
if __name__ == '__main__':
    #model
    recognizer = HandwrittenCharactersRecognizer()
    #recognizer.load_data()
    #recognizer.normalize_data()
    #recognizer.build_model(show=True)
    #recognizer.fit(save=True)
    #accuracy = recognizer.test()
    #print("Accuracy: %.2f%%" % (accuracy))
    recognizer.load_model()
    
    #test
    tester = OwnImageTester(path_to_img='c:/users/pznam/desktop/imgs/img7.jpg')
    tester.normalize()
    char = tester.test(model=recognizer.model, with_plot=True)
    print(char)