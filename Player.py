import random
import copy
import numpy as np
from State import State

class Player:
    def __init__(self, name, env, pieces, EXP_DECAY_RATE, explore=True):
        self.LEARNING_RATE = 0.2
        self.GAMMA = 0.9
        self.exploration_rate = 0.9 if explore else 0
        self.EXP_DECAY_RATE = EXP_DECAY_RATE
        
        self.name = name
        
        self.env = env
        
        self.states = []
        self.states_values = {}
        
        self.pieces = pieces
        self.orig_pieces = copy.deepcopy(pieces)
        self.pieces_on_board = []
        
        self.win = 0
        self.loss = 0
        self.draw = 0
    
    def choose_action(self, state, symbol):
        action = None
        moves = self.env.get_legal_moves(self)

        if random.random() < self.exploration_rate:
            try:
                action = random.choice(moves)
            except IndexError:
                return {'destination':(0, 0), 'size':-1}
        else:
            # use our states-values mappings to choose the best move

            max_value = -999
            get_val = self.states_values.get
            for move in moves:
                next_state = State(state.board.copy())
                next_state.board[move['destination'][0], move['destination'][1]] = symbol * move['size']
                if len(move['origin']) == 2:
                    # if the pieces was already on the board, fix it
                    next_state.board[move['origin'][0], move['origin'][1]] = 0
                    
                val = get_val(next_state)
                value = 0 if val is None else val

                if value > max_value:
                    max_value = value
                    action = move
            
        return action
    
    def update_values(self, reward):
        get_val = self.states_values.get
        new_states_values = {}
        for state in reversed(self.states):
            value = get_val(state)
            if value is None:
                value = 0

            value += self.LEARNING_RATE * (self.GAMMA * reward - value)
            new_states_values.update({state: value})
            reward = float(value)
        
        self.states_values.update(new_states_values)
        self.decay_exploration_rate()

    def decay_exploration_rate(self):
        self.exploration_rate *= self.EXP_DECAY_RATE
    
    def reset(self):
        self.states = []
        self.pieces = copy.deepcopy(self.orig_pieces)
        self.pieces_on_board = []
        
    def save_policy(self, time):
        fout = open("policy_{}".format(self.name), 'w')
        fout.write("{}\n".format(time))
        for state, value in self.states_values.items():
            fout.write("{};{}\n".format(state, value))
            
        fout.close()
    
    def load_policy(self, name):
        lines = []
        with open(name, 'r') as fin:
            lines = fin.read().split('\n')
        
        for line in lines[1:]:
            try:
                state, value = line.split(';')
            except ValueError:
                continue
            # board = self.str_to_list(state)
            #st = State(board)
            self.states_values[state] = float(value)

    def __str__(self):
        return "{}: {}".format(self.name, self.pieces)