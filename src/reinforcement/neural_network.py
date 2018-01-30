from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np


class NeuralNetwork(object):
    def __init__(self):
        self.model = Sequential()

    def train(self, x, y):
        # self.model.fit(np.array([x]), np.array([y]), epochs=100, verbose=0)
        for _ in range(100):
            print(self.model.train_on_batch(np.array([x]), np.array([y])), end=" ")
        print()

    def predict(self, x):
        return self.model.predict(np.array([x]))[0]


class EffectOfAction(NeuralNetwork):
    def __init__(self, input_size, hidden_layer_size, output_size):
        NeuralNetwork.__init__(self)
        self.model.add(Dense(hidden_layer_size, input_dim=input_size))
        self.model.add(Activation("sigmoid"))
        self.model.add(Dense(output_size))
        self.model.compile(optimizer="sgd", loss="mean_squared_error")


class RewardForAction(NeuralNetwork):
    def __init__(self, input_size, hidden_layer_size):
        NeuralNetwork.__init__(self)
        self.model.add(Dense(hidden_layer_size, input_dim=input_size))
        self.model.add(Activation("sigmoid"))
        self.model.add(Dense(1))
        self.model.compile(optimizer="sgd", loss="mean_squared_error")


class SuccessOfAction(NeuralNetwork):
    def __init__(self, input_size, hidden_layer_size):
        NeuralNetwork.__init__(self)
        self.model.add(Dense(hidden_layer_size, input_dim=input_size))
        self.model.add(Activation("sigmoid"))
        self.model.add(Dense(1))
        self.model.compile(optimizer="sgd", loss="mean_squared_error")
