"""
Train a supervised CNN model using optimal stock as label
"""
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from ..base_model import BaseModel

from utils.data import normalize


class StockLSTM(BaseModel):
    def __init__(self, num_classes, window_length, weights_file='weights/lstm.h5'):
        self.model = None
        self.weights_file = weights_file
        self.num_classes = num_classes
        self.window_length = window_length

    def build_model(self, load_weights=True):
        """ Load training history from path

        Args:
            load_weights (Bool): True to resume training from file or just deploying.
                                 Otherwise, training from scratch.

        Returns:

        """
        if load_weights:
            self.model = keras.models.load_model(self.weights_file)
            print('Successfully loaded model')
        else:
            self.model = Sequential()
            self.model.add(keras.layers.LSTM(20, input_shape=(self.num_classes, self.window_length)))
            self.model.add(Dense(64))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(self.num_classes, activation='softmax'))

            self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
            print('Built model from scratch')
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

    def train(self, X_train, Y_train, X_val, Y_val, verbose=True):
        #continue_train = True
        for episodes in range(1,21):
            print('Episode ', episodes)
            #while continue_train:
            self.model.fit(X_train, Y_train, batch_size=128, epochs=50, validation_data=(X_val, Y_val),
                           shuffle=True, verbose=verbose)
            #save_weights = input('Type True to save weights\n')
            #if save_weights:
            #    self.model.save(self.weights_file)
            #continue_train = input("True to continue train, otherwise stop training...\n")
        print('lstm model saved.')
        self.model.save(self.weights_file)
        print('Finish.')

    def evaluate(self, X_test, Y_test, verbose=False):
        return self.model.evaluate(X_test, Y_test, verbose=verbose)

    def predict(self, X_test, verbose=False):
        return self.model.predict(X_test, verbose=verbose)

    def predict_single(self, observation):
        """ Predict the action of a single observation

        Args:
            observation: (num_stocks + 1, window_length)

        Returns: a single action array with shape (num_stocks + 1,)

        """
        action = np.zeros((self.num_classes,))
        obsX = observation[:, -self.window_length:, 3] / observation[:, -self.window_length:, 0]
        obsX = normalize(obsX)
        obsX = np.expand_dims(obsX, axis=0)
        with self.graph.as_default():
            current_action_index = self.model.predict_classes(obsX, verbose=False)
        action[current_action_index] = 1.0
        return action
