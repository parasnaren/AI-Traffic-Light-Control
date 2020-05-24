import sys
import numpy as np
import keras.backend as K
import os
import logging
logging.basicConfig(level=logging.INFO)

from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, LSTM, Lambda
from keras.regularizers import l2

class Agent:
    """ Agent Class (Network) for DDQN
    """

    def __init__(self, input_dim, output_dim, lr, tau, dueling, config, export_path):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tau = tau
        self.dueling = dueling
        self.lr = lr

        self.width = config.get('width_layers', 500)
        self.num_layers = config.get('num_layers', 4)
        self.epochs = config.get('training_epochs')
        self.model_path = config.get('model_path')
        self.export_path = export_path

        # Initialize Deep Q-Network
        self.model = self.network()

        # Build target Q-Network
        self.target_model = self.network()
        self.target_model.set_weights(self.model.get_weights())


    def huber_loss(self, y_true, y_pred):
        return K.mean(K.sqrt(1 + K.square(y_pred - y_true)) - 1, axis=-1)


    def network(self):
        """ Build Deep Q-Network
        """
        # Return model if exists for continued training/for testing
        if self.model_path:
            logging.info('Loaded model: {}'.format(self.export_path))
            return load_model(self.export_path)

        logging.info('Creating model')
        model = Sequential()
        model.add(Dense(self.width, input_shape=(self.input_dim,)))
        for _ in range(self.num_layers):
            model.add(Dense(self.width, activation='relu'))

        if(self.dueling):
            # Have the network estimate the Advantage function as an intermediate layer
            model.add(Dense(self.output_dim + 1, activation='linear'))
            model.add(Lambda(lambda i: K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True), output_shape=(self.output_dim,)))
        else:
            model.add(Dense(self.output_dim, activation='linear'))

        model.compile(Adam(self.lr), 'mse')
        return model


    def transfer_weights(self):
        """ Transfer Weights from Model to Target at rate Tau
        """
        W = self.model.get_weights()
        tgt_W = self.target_model.get_weights()
        for i in range(len(W)):
            tgt_W[i] = self.tau * W[i] + (1 - self.tau) * tgt_W[i]
        self.target_model.set_weights(tgt_W)


    def fit(self, inp, targ):
        """ Perform training
        """
        self.model.fit(inp, targ, epochs=1, verbose=0)


    def predict(self, inp):
        """ Q-Value Prediction
        """
        return self.model.predict(inp)


    def target_predict(self, inp):
        """ Q-Value Prediction (using target network)
        """
        return self.target_model.predict(inp)


    def save_model(self, path):
        if(self.dueling):
            path += '_dueling'
        self.model.save(path + '.h5')


    def load_model(self, path):
        self.model = load_model(path)
