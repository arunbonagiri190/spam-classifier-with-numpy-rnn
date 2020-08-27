import numpy as np
import pandas as pd
import pickle
from pathlib import Path

class DataLoader:

    path = '../data/'
    test_data = pd.DataFrame([])

    # function for loading data from disk
    @classmethod
    def load_data(self):
        """
        this function is responsible for loading data from disk.
        
        Parameters:
            (no-parameters)

        Returns:
            X   :   numpy array         
            y   :   numpy array         
        
        """
        
        if(not Path(self.path+'encoded_spam.pkl').is_file()):
            print("[util]: cleaned_data not found at '",self.path,"'")
            #quit()

        print("[util]: Loading '",self.path+'encoded_spam.pkl',"'")
        data = pd.read_pickle(self.path+'encoded_spam.pkl')
        data = data.sample(frac=1).reset_index(drop=True)
        data = data.head(90).copy()
        self.test_data = data.tail(10).copy().reset_index(drop=True)

        X = np.array(data['encoded_text'])
        y = np.array(data[['ham', 'spam']])

        return X, y
    
    @classmethod
    def load_test_data(self):
        """
        this function is responsible for loading test data.

        Parameters:
            (no-parameters)

        Returns:
            text:   pandas series
            X   :   numpy array         
            y   :   numpy array         
        
        """

        text = self.test_data['text']
        X = np.array(self.test_data['encoded_text'])
        y = np.array(self.test_data[['ham', 'spam']])

        return text, X, y


class Activations:
    
    # sigmoid activation function with derivative
    @classmethod
    def sigmoid(self, x, derivative=False):
        if(derivative):
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        
        return 1.0/(1.0 + np.exp(-x))


    # relu activation function with derivative
    @classmethod
    def relu(self, x, derivative=False):
        if(derivative):
            return x > 0
        
        return np.maximum(x, 0)


    # softmax activation function
    @classmethod
    def softmax(self, z):
        exp = np.exp(z)
        return exp / sum(exp)