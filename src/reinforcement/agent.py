from reinforcement import action


class Agent(object):
    def __init__(self, symbol, environment):
        self.symbol = str(symbol)
        self.environment = environment
        self.vision_field_radius = 2
        self.row, self.col = self.find_coordinates(self.symbol, str(environment))
        self.actions = [
            action.MoveUp(self.environment, self),
            action.MoveDown(self.environment, self),
            action.MoveLeft(self.environment, self),
            action.MoveRight(self.environment, self),
            action.Stay(self.environment, self),
        ]

    def find_coordinates(self, symbol, environment_str):
        for i, row in enumerate(environment_str.split()):
            location = row.find(symbol)
            if location != -1:
                return i, location
        raise Exception

    def get_visible_field_coordinates(self):
        return (
            self.row - self.vision_field_radius,
            self.row + self.vision_field_radius + 1,
            self.col - self.vision_field_radius,
            self.col + self.vision_field_radius + 1,
        )

    def move_up(self):
        self.row, self.col = self.actions[0].perform()

    def move_down(self):
        self.row, self.col = self.actions[1].perform()

    def move_left(self):
        self.row, self.col = self.actions[2].perform()

    def move_right(self):
        self.row, self.col = self.actions[3].perform()

    def move_stay(self):
        self.row, self.col = self.actions[4].perform()
