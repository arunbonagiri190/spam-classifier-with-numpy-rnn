import numpy as np
from numpy.random import randn
from util import Activations

class RNN:
    
    input_size = 750
    output_size = 2
    hidden_size = 90
    
    lr = 0.001
    cr = 1
    
    def __init__(self):
        
        self.weights = { 'input-to-hidden' :np.random.randn(self.hidden_size, self.input_size) / 1000, 
                         'hidden-to-hidden':np.random.randn(self.hidden_size, self.hidden_size) / 1000,
                         'hidden-to-output':np.random.randn(self.output_size, self.hidden_size) / 1000 }
        
        self.biases = { 'hidden':np.zeros((self.hidden_size, 1)),
                        'output':np.zeros((self.output_size, 1)) }
        
        print("[RNN]: Intializing ... ")
        print("Weights\t input-to-hidden:",self.weights['input-to-hidden'].shape,
              "  hidden-to-hidden:",self.weights['hidden-to-hidden'].shape,
              "  hidden-to-output:",self.weights['hidden-to-output'].shape)
        print("Biases\t\t\t\t\thidden:",self.biases['hidden'].shape,"  \t\toutput:",self.biases['output'].shape,"\n")
    
    def fit(self, X, y, ephocs=1):
        """
        function for training the network

        Parameters:
            X       :   numpy array
            y       :   numpy array
            ephocs  :   int

        Returns:
            (no-returns)
        """
        
        batch_size = X.shape[0]
        print('[RNN]: Training on ',ephocs, 'ephocs with batch_size:', batch_size)

        for epoch in range(ephocs):
            
            total_loss = []
            
            for i in range(batch_size):

                # forward
                out = self.__forward__(X[i])

                # loss
                total_loss.append(-np.log(out[np.argmax(y[i])])[0])
                
                # backward
                out[np.argmax(y[i])] -= 1
                self.__backward__(out)
            
            if((epoch+1)%1000 == 0):
                print('\tat epoch ',(epoch+1),'\t...   loss: ',sum(total_loss)/batch_size)

    def predict(self, inputs):
        """
        function for predicting label for given inputs

        Parameters:
            inputs       :   numpy array

        Returns:
            return's probabilities, in the form of  numpy array
        """
        
        hidden_activations = np.zeros((self.hidden_size, 1))
        
        for _ , x in enumerate(inputs):
            hidden_activations = np.tanh( self.weights['input-to-hidden'].dot(x) + self.weights['hidden-to-hidden'].dot(hidden_activations) + self.biases['hidden'])
        
        return Activations.softmax(self.weights['hidden-to-output'].dot(hidden_activations) + self.biases['output'])
    
    
    def __forward__(self, inputs):
        """
        an implementation of forward propagation of a network

        Parameters:
            inputs       :   numpy array

        Returns:
            outputs       :   numpy array
        """
        
        hidden_activations = np.zeros((self.hidden_size, 1))
        
        # saving to history
        self.last_inputs = inputs
        self.states = { 0: hidden_activations }

        # passing each word into input and hidden layers
        for i, x in enumerate(inputs):
            input_z = self.weights['input-to-hidden'].dot(x)
            hidden_activations = np.tanh( input_z + self.weights['hidden-to-hidden'].dot(hidden_activations) + self.biases['hidden'])
            self.states[i + 1] = hidden_activations
        
        # passing through output layer
        outputs = Activations.softmax(self.weights['hidden-to-output'].dot(hidden_activations) + self.biases['output'])
        
        return outputs
    
    def __backward__(self, error):
        """
        an implementation of backward propagation of a network

        Parameters:
            error       :   numpy array

        Returns:
            (no-returns)
        """
        
        # intializing new weights and biases for input layer, hidden layer
        nw_input_to_hidden = np.zeros((self.hidden_size, self.input_size))
        nw_hidden_to_hidden = np.zeros((self.hidden_size, self.hidden_size))
        nb_hidden = np.zeros((self.hidden_size, 1))
        
        # Calculating outputlayer new weights and biases
        nw_hidden_to_output = error.dot(self.states[len(self.last_inputs)].T)
        nb_output = error
        
        # calculating hidden layer error with respect to output layer
        hidden_error = self.weights['hidden-to-output'].T.dot(error)
        
        # calculating new weights for hidden layers
        for ti in reversed(range(len(self.last_inputs))):
            
            temp = ((1 - self.states[ti + 1] ** 2) * hidden_error)

            # Calculating new hiddenlayer bias
            nb_hidden += temp
            
            # Calculating new hidden-to-hidden weights
            nw_hidden_to_hidden += temp.dot(self.states[ti].T)
            
            # Calculating new input-to-hidden weights
            nw_input_to_hidden += temp.dot(self.last_inputs[ti].T)
            
            # updating hidden layer error
            hidden_error = self.weights['hidden-to-hidden'].dot(temp)
        
        # Clipping all gradients
        for d in [nw_input_to_hidden, nw_hidden_to_hidden, nw_hidden_to_output, nb_hidden, nb_output]:
            np.clip(d, -self.cr, self.cr, out=d)
        
        # updating weights and biases
        self.weights['hidden-to-output'] -= self.lr * nw_hidden_to_output
        self.weights['hidden-to-hidden'] -= self.lr * nw_hidden_to_hidden
        self.weights['input-to-hidden'] -= self.lr * nw_input_to_hidden
        
        self.biases['output'] -= self.lr * nb_output
        self.biases['hidden'] -= self.lr * nb_hidden
