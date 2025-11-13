# model_rl.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
import numpy as np

class Model:
    def __init__(self, ninput, layers, n_actions, model=None):
        """
        ninput: numero di input (celle dello stato)
        layers: lista con il numero di neuroni per layer nascosto
        n_actions: numero di azioni possibili
        """
        self.keras_model = model or self.build_model(ninput, layers, n_actions)

    def build_model(self, ninput, layers, n_actions):
        input_layer = Input(shape=(ninput,))
        x = input_layer
        for n in layers:
            x = Dense(n, activation='relu')(x)
        output_layer = Dense(n_actions, activation='linear')(x)  # Q-values

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mse', optimizer='rmsprop')
        return model

    def predict(self, states):
        """
        states: lista di stati o singolo stato
        restituisce array NumPy con Q-values
        """
        states = np.array(states)
        if len(states.shape) == 1:  # singolo stato
            states = np.expand_dims(states, axis=0)
        return self.keras_model.predict(states, verbose=0)

    def fit(self, x, y, epochs=1, batch_size=32, verbose=0):
        """
        Funzione wrapper per allenare il modello
        x: array di stati
        y: array di target Q-values
        """
        self.keras_model.fit(np.array(x), np.array(y),
                             epochs=epochs,
                             batch_size=batch_size,
                             verbose=verbose)
