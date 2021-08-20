from recognizer import HandwrittenCharactersRecognizer
from tester import OwnImageTester


    
    
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
    tester = OwnImageTester(path_to_img='c:/users/user/desktop/m.png')
    tester.normalize()
    char = tester.test(model=recognizer.model, with_plot=True)
    print(char)