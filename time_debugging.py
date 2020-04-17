import timeit

SETUP = """
import time
from copy import deepcopy
from Environment import Environment
from State import State
from Player import Player

DECAY_RATE = .99985

env = Environment(4, 4, 4)
p1 = Player('p1', env, [[{'location':j, 'size':4-j} for j in range(4)] for i in range(3)], DECAY_RATE, False)
p2 = Player('p2', env, [[{'location':j, 'size':4-j} for j in range(4)] for i in range(3)], DECAY_RATE, False)
p1.choose_action(env.state, env.turn)
"""

STMT = "next_state, result = env.update(action, player)"

timer = timeit.Timer(stmt = STMT, setup = SETUP)
num = 1000000
print(timeit.timeit(number = num)/num)
