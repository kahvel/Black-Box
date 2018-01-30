from reinforcement import neural_network


class Action(object):
    def __init__(self, environment, agent, d_row, d_col):
        self.agent = agent
        self.environment = environment
        self.d_row = d_row
        self.d_col = d_col
        self.field_of_vision = self.agent.vision_field_radius * 2 + 1
        self.environment_network = neural_network.EffectOfAction(self.field_of_vision ** 2, self.field_of_vision ** 2, self.field_of_vision ** 2)
        self.reward_network = neural_network.RewardForAction(self.field_of_vision ** 2, self.field_of_vision ** 2)
        self.success_network = neural_network.SuccessOfAction(self.field_of_vision ** 2, self.field_of_vision ** 2)

    def perform(self):
        new_row = self.agent.row + self.d_row
        new_col = self.agent.col + self.d_col
        environment_before_action = self.environment.get_area_list(*self.agent.get_visible_field_coordinates())
        if self.environment.can_move_agent(new_row, new_col):
            reward = self.environment.calculate_reward(new_row, new_col)
            self.reward_network.train(environment_before_action, reward)
            self.success_network.train(environment_before_action, 1)
            self.environment.move_agent(self.agent.row, self.agent.col, new_row, new_col, self.agent.symbol)
            environment_after_action = self.environment.get_area_list(*self.agent.get_visible_field_coordinates())
            self.environment_network.train(environment_before_action, environment_after_action)
            return new_row, new_col
        else:
            self.success_network.train(environment_before_action, 0)
            return self.agent.row, self.agent.col

    # def numbers_matrix_to_string(self, matrix):
    #     return "\n".join(" ".join(map(str, row)) for row in matrix)
    #
    # def list_to_matrix(self, list):
    #     return [[list[i*self.field_of_vision + j] for j in range(self.field_of_vision)] for i in range(self.field_of_vision)]

    def predict_environment(self):
        current_environment = self.environment.get_area_list(*self.agent.get_visible_field_coordinates())
        predicted_environment = self.environment_network.predict(current_environment)
        return predicted_environment

    def predict_reward(self):
        current_environment = self.environment.get_area_list(*self.agent.get_visible_field_coordinates())
        predicted_reward = self.reward_network.predict(current_environment)
        return predicted_reward
