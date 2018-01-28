

class Environment(object):
    def __init__(self, environment_str):
        self.environment_matrix = self.string_to_numbers_matrix(environment_str)
        self.initial_matrix = self.environment_matrix[:]
        self.n_row = len(self.environment_matrix)
        self.n_col = len(self.environment_matrix[0])
        self.cell_count = self.n_row * self.n_col

    def string_to_numbers_matrix(self, environment_as_str):
        return [list(map(ord, row)) for row in environment_as_str.split()]

    def matrix_to_list(self, matrix):
        return [element for row in matrix for element in row]

    def list_to_matrix(self, list):
        return [[list[i*self.n_col + j] for j in range(self.n_col)] for i in range(self.n_row)]

    def numbers_matrix_to_string(self, matrix):
        return "\n".join("".join(map(chr, row)) for row in matrix)

    def get_environment_list(self):
        return self.matrix_to_list(self.environment_matrix)

    def get_field_of_vision(self, agent):
        row1, row2, col1, col2 = agent.get_visible_field_coordinates()
        return self.matrix_to_list(map(lambda x: x[col1: col2], self.environment_matrix[row1: row2]))

    def in_bounds(self, new_row, new_col):
        return 0 <= new_row < self.n_row and 0 <= new_col < self.n_col

    def not_blocked(self, new_row, new_col):
        return self.environment_matrix[new_row][new_col] == ord(".")

    def can_move_agent(self, new_row, new_col):
        return self.in_bounds(new_row, new_col) and self.not_blocked(new_row, new_col)

    def get_agents_new_coordinates(self, old_row, old_col, new_row, new_col):
        if self.can_move_agent(new_row, new_col):
            return new_row, new_col
        else:
            return old_row, old_col

    def move_agent(self, agent, d_row, d_col):
        new_row, new_col = self.get_agents_new_coordinates(agent.row, agent.col, agent.row + d_row, agent.col + d_col)
        self.environment_matrix[agent.row][agent.col] = ord(".")  # self.initial_matrix[agent.row][agent.col]
        self.environment_matrix[new_row][new_col] = ord(agent.symbol)
        return new_row, new_col

    def __repr__(self):
        return self.numbers_matrix_to_string(self.environment_matrix)



