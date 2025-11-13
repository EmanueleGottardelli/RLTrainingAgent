import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
import numpy as np


class Model:
    def __init__(self, ninput, layers, model=None):
        self.keras_model = model or self.build_model(ninput, layers)

    def build_model(self, ninput, layers):
        input_layer = Input(shape=(ninput,))
        x = input_layer
        for n in layers:
            x = Dense(n, activation='relu',)(x)
        output_layer = Dense(3, activation='softmax')(x)

        model = tf.keras.Model(input_layer, output_layer)

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        return model

    def train(self, plays, split_ratio=0.2, epochs=1, batch_size=128):
        features, targets = self.preprocess(plays)

        idx = int(split_ratio*len(features))

        train_X, train_Y = features[idx:], targets[idx:]
        test_X, test_Y = features[:idx], targets[:idx]

        history = self.keras_model.fit(
            train_X, train_Y,
            validation_data=(test_X, test_Y),
            epochs=epochs,
            batch_size=batch_size)
        print(history)

    def preprocess(self, plays):
        print(f'Proprocessing {len(plays)} plays...')
        dataset = []
        for states, winner in plays:
            if winner == 0:
                continue
            rows = [(move.state.cells, winner) for move in states]
            if states[-1].state.winner() != 0:
                rows.extend(self._preprocess_critical_action(states, winner))

            dataset.extend(rows)

        np.random.shuffle(dataset)

        features, targets = tuple(np.array(e) for e in zip(*dataset))
        targets = tf.keras.utils.to_categorical(targets, num_classes=3)

        return features, targets

    def _preprocess_critical_action(self, states, winner):
        critical_state = states[-3].state
        critical_action = states[-2].action
        data = []
        for action in critical_state.actions():
            state = critical_state.move(action)
            if action != critical_action:
                data.append((state.cells, critical_state.player()))

        return data

    def predict(self, states):
        return self.keras_model.predict(states)