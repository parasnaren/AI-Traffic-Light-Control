import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam


class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, model_file_path, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model_file_path = model_file_path
        self._model = self._build_model(num_layers, width)


    def _build_model(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network
        """
        if os.path.exists(self._model_file_path):
            model = load_model(self._model_file_path)
            print('Loading Keras model..')
        else:
            # inputs = keras.Input(shape=(self._input_dim,))
            model = Sequential()
            model.add(LSTM(200, input_shape=(1, self._input_dim), return_sequences=False))
            for _ in range(num_layers):
                model.add(Dense(width, activation='relu'))
            model.add(Dense(self._output_dim, activation='linear'))
            model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))
            print('Creating new Keras model..')
            print(model.summary())

        return model
    

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = state.reshape(-1, 1, self._input_dim)
        # state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        states = states.reshape(states.shape[0], -1, self._input_dim)
        return self._model.predict(states)


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        states = states.reshape(states.shape[0], -1, self._input_dim)
        self._model.fit(states, q_sa, epochs=1, verbose=0)


    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        model_path = os.path.join(path, 'trained_model.h5')
        if os.path.exists(model_path):
            os.remove(model_path)
        self._model.save(os.path.join(path, 'trained_model.h5'))
        # plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path, model_type):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)
        self._model_type = model_type


    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        print(model_folder_path)
        if 'model_0' not in model_folder_path:
            model_file_path = os.path.join(model_folder_path, 'trained_model.h5')
            loaded_model = load_model(model_file_path)
            return loaded_model
        return None


    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        if self._model_type == 'dqn-lstm':
            state = state.reshape(-1, 1, self._input_dim)
        elif self._model_type == 'dqn':
            state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    @property
    def input_dim(self):
        return self._input_dim