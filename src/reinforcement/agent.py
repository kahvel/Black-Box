from reinforcement import action, memory


class Agent(object):
    def __init__(self, symbol, environment):
        self.symbol = str(symbol)
        self.environment = environment
        self.vision_field_radius = 2
        self.row, self.col = self.find_coordinates(self.symbol, str(environment))
        self.memory = memory.Memory(4)
        self.actions = [
            action.Action(self.environment, self, -1, 0),
            action.Action(self.environment, self, 1, 0),
            action.Action(self.environment, self, 0, -1),
            action.Action(self.environment, self, 0, 1),
            action.Action(self.environment, self, 0, 0),
        ]

    def find_coordinates(self, symbol, environment_str):
        for i, row in enumerate(environment_str.split()):
            location = row.find(symbol)
            if location != -1:
                return i, location
        raise Exception

    def get_square_coordinates(self, row, col, radius):
        return (
            row - radius,
            row + radius + 1,
            col - radius,
            col + radius + 1,
        )

    def get_visible_field_coordinates(self):
        return self.get_square_coordinates(self.row, self.col, self.vision_field_radius)

    def perform_action(self, action_id):
        self.memory.save(self.environment.get_area_list(*self.get_visible_field_coordinates()), self.actions[action_id])
        self.row, self.col = self.actions[action_id].perform()

    def move_up(self):
        self.perform_action(0)

    def move_down(self):
        self.perform_action(1)

    def move_left(self):
        self.perform_action(2)

    def move_right(self):
        self.perform_action(3)

    def move_stay(self):
        self.perform_action(4)
