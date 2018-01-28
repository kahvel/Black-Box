from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np


class Action(object):
    def __init__(self, environment, agent):
        self.agent = agent
        self.environment = environment
        self.d_row = None
        self.d_col = None
        self.field_of_vision = self.agent.vision_field_radius * 2 + 1
        self.nn = Sequential()
        self.nn.add(Dense(self.field_of_vision**2, input_dim=self.field_of_vision**2))
        self.nn.add(Activation("sigmoid"))
        self.nn.add(Dense(self.field_of_vision ** 2))
        self.nn.compile(optimizer="sgd", loss="mean_squared_error")

    def perform(self):
        environment_before_action = self.environment.get_field_of_vision(self.agent)
        new_row, new_col = self.environment.move_agent(self.agent, self.d_row, self.d_col)
        environment_after_action = self.environment.get_field_of_vision(self.agent)
        # print(np.array([environment_before_action]))
        # print(np.array([environment_after_action]))
        self.nn.fit(np.array([environment_before_action]), np.array([environment_after_action]), epochs=100, verbose=0)
        # print(self.nn.train_on_batch(np.array([environment_before_action]), np.array([environment_after_action])))
        return new_row, new_col

    def numbers_matrix_to_string(self, matrix):
        return "\n".join(" ".join(map(str, row)) for row in matrix)

    def list_to_matrix(self, list):
        return [[list[i*self.field_of_vision + j] for j in range(self.field_of_vision)] for i in range(self.field_of_vision)]

    def predict(self):
        current_environment = self.environment.get_field_of_vision(self.agent)
        predicted_environment = self.nn.predict(np.array([current_environment]))[0]
        return predicted_environment


class MoveUp(Action):
    def __init__(self, environment, agent):
        Action.__init__(self, environment, agent)
        self.d_row = -1
        self.d_col = 0


class MoveDown(Action):
    def __init__(self, environment, agent):
        Action.__init__(self, environment, agent)
        self.d_row = 1
        self.d_col = 0


class MoveLeft(Action):
    def __init__(self, environment, agent):
        Action.__init__(self, environment, agent)
        self.d_row = 0
        self.d_col = -1


class MoveRight(Action):
    def __init__(self, environment, agent):
        Action.__init__(self, environment, agent)
        self.d_row = 0
        self.d_col = 1


class Stay(Action):
    def __init__(self, environment, agent):
        Action.__init__(self, environment, agent)
        self.d_row = 0
        self.d_col = 0
