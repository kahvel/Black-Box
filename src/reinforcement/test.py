from reinforcement.environment import Environment
from reinforcement.agent import Agent

import random


environment_as_str = """
MMMMMMMMMMMMMM
MMMMMMMMMMMMMM
MM*........*MM
MM.....1....MM
MM*......*..MM
MM....*.....MM
MM..........MM
MM..........MM
MM.......*..MM
MM..*.......MM
MM......*...MM
MM*........*MM
MMMMMMMMMMMMMM
MMMMMMMMMMMMMM
"""[1:-1]

environment = Environment(environment_as_str)
agent = Agent(1, environment)

mapping = {"w": 0, "s": 1, "a": 2, "d": 3}
while True:
    # action = random.randint(0, 4)
    print(environment)
    action = input()
    if action.islower():
        if action in mapping:
            agent.perform_action(mapping[action])
    else:
        if action.lower() in mapping:
            action = mapping[action.lower()]
            print(agent.actions[action].predict_environment())
            print(agent.actions[action].predict_reward())

    # print(agent.row, agent.col, action)
    # input()














