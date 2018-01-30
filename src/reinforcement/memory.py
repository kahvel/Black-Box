import reinforcement


class FixedSizeList(object):
    def __init__(self, size):
        self.size = size
        self.elements = [None for _ in range(self.size)]  # type: list[reinforcement.action.Action]
        self.counter = 0

    def add(self, element):
        if self.counter < self.size:
            self.elements[self.counter] = element
            self.counter += 1
        else:
            del self.elements[0]
            self.elements.append(element)

    def get_last(self):
        return self.elements[self.counter - 1]


class Memory(object):
    def __init__(self, size):
        self.size = size
        self.environment = FixedSizeList(self.size)
        self.action = FixedSizeList(self.size)

    def save(self, environment, action):
        self.environment.add(environment)
        self.action.add(action)
