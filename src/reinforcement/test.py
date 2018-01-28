from reinforcement.environment import Environment
from reinforcement.agent import Agent

import random


environment_as_str = """
MMMMMMMMMMMMMM
MMMMMMMMMMMMMM
MM..........MM
MM.....1....MM
MM..........MM
MM..........MM
MM..........MM
MM..........MM
MM..........MM
MM..........MM
MM..........MM
MM..........MM
MMMMMMMMMMMMMM
MMMMMMMMMMMMMM
"""[1:-1]

environment = Environment(environment_as_str)
agent = Agent(1, environment)

while True:
    action = random.randint(0, 4)
    print(environment)
    print(agent.actions[action].predict())
    agent.row, agent.col = agent.actions[action].perform()
    # print(agent.row, agent.col, action)
    input()














