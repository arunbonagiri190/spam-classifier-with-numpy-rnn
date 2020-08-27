import rnn
from util import DataLoader, Activations
import numpy as np

def main():
    
    # loading data
    X, y = DataLoader.load_data()
    text, X_test, y_test = DataLoader.load_test_data()

    # intializing network
    network = rnn.RNN()

    network.fit(X, y, ephocs=3000)

    # sample predictions
    label_names = ['ham', 'spam']
    tag = np.random.randint(0,10)
    print('\n\n[main]: Test prediction:',
          '\nSMS: ',text[tag] ,
          '\n\nactual: ',label_names[np.argmax(y_test[tag])],
          '\nprediction: ',label_names[np.argmax(network.predict(X_test[tag]))])
    
    tag = np.random.randint(0,10)
    print('\n\n[main]: Test prediction:',
          '\nSMS: ',text[tag] ,
          '\n\nactual: ',label_names[np.argmax(y_test[tag])],
          '\nprediction: ',label_names[np.argmax(network.predict(X_test[tag]))])


if __name__ == "__main__":
    main()