from emnist import extract_training_samples, extract_test_samples
from tensorflow.keras import Sequential, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow import nn




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