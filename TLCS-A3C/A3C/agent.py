import numpy as np
from keras.optimizers import RMSprop

class Agent:
    """ Agent Generic Class
    """

    def __init__(self, input_dim, output_dim, lr, tau=0.001, epochs=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tau = tau
        self.rms_optimizer = RMSprop(lr=lr, epsilon=0.1, rho=0.99)
        self.epochs = epochs

    def fit(self, state, target):
        """ Perform training
        """
        self.model.fit(state, target, epochs=self.epochs, verbose=0)

    def predict(self, state):
        """ Critic Value Prediction
        """
        return self.model.predict(state)
