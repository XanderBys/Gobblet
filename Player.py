import random
import pickle
import math
import copy
import numpy as np
from State import State
from Model import Model
from Memory import Memory
from Piece import Piece

class Player:
    def __init__(self, name, env, symbol, memory_capacity, model=None, BATCH_SIZE=0, EPSILON_ARGS=(0,0,0), maximize_entropy=False, use_PER=False, PER_hyperparams=(0,0,0)):
        self.GAMMA = 0.9
        self.EPSILON_MAX, self.EPSILON_MIN, self.LAMBDA = EPSILON_ARGS
        self.exploration_rate = self.EPSILON_MAX
        self.BATCH_SIZE = BATCH_SIZE
        self.TEST_SPLIT = 0.9
        self.epsilon_decay_steps = 0
        
        self.name = name
        self.env = env
        self.symbol = symbol
        self.samples = []
        self.model = model
        self.targets = Model(self.model.num_states, self.model.num_actions, dueling=self.model.dueling) if model is not None else None
        self.use_PER = use_PER
        self.PER_hyperparams = PER_hyperparams
        self.memory = Memory(memory_capacity, False, self.use_PER, self.PER_hyperparams)
        self.loc_accuracy = []
        self.piece_accuracy = []
        self.loc_loss = []
        self.piece_loss = []
        self.total_rewards = [0]
        self.average_reward = [0]
        self.invalid_moves = [0]
        self.regret = [1] # optimal reward - actual reward
        self.was_random = False
        self.maximize_entropy = maximize_entropy

        self.pieces = []
        self.create_pieces()
        
        self.win = 0
        self.losses = 0
        self.draw = 0
    
    def choose_action(self, state):
        action = None
        # choose an action takes two parts:
        # one for choosing location and another for choosing the piece
        if random.random() < self.exploration_rate:
            location = (random.choice(range(4)), random.choice(range(4)))
            piece = random.choice(self.pieces)
        else:
            # exploit
            q_values = self.model.predict_one(state.board.reshape(-1))
            
            max_loc = np.amax(q_values[0])
            # convert from linear to 2D indicies
            location = np.where(q_values[0][0]==max_loc)[0][0]
            location = (location // self.env.NUM_COLS, location % self.env.NUM_COLS)
            
            piece = self.pieces[np.argmax(q_values[1])] 
        
        return (piece, location)
    
    def train(self):
        # train the model based on the reward
        flatten = lambda arr: arr.reshape(-1) if arr is not None else np.zeros(self.env.NUM_ROWS*self.env.NUM_COLS)
        if self.use_PER:
            tree_idxs, samples = self.memory.sample(self.BATCH_SIZE)
        else:
            samples = self.memory.sample(self.BATCH_SIZE)
        
        if len(samples) == 0:
            return
        
        states, actions, rewards, next_states, completes = np.array(samples).T

        states = np.array(list(map(lambda x: flatten(x.board), states)))
        next_states = np.array(list(map(lambda x: flatten(x.board) if x is not None else np.zeros(self.env.NUM_ROWS*self.env.NUM_COLS), next_states)))
        
        q_s_a = self.targets.predict_batch(states)
        q_s_a_p = self.model.predict_batch(next_states)

        # training arrays
        x = np.array(list(map(flatten, states)))
        y = [np.array(list(map(flatten, q_s_a[0]))), np.array(list(map(flatten, q_s_a[1])))]
        
        actions = [np.squeeze(np.array(list(map(lambda x: None if x is None else x[1][0] * self.env.NUM_COLS + x[1][1], actions)))),
                   np.squeeze(np.array(list(map(lambda x: None if x is None else x[0].idx, actions))))]
        num_actions = tuple(map(lambda x: 0 if x.shape == () else range(len(x)), actions))
        next_actions = list(map(lambda x: np.argmax(x, axis=1), q_s_a_p))
        fake_states = next_states.copy()
        fake_states[range(len(next_actions[0])), next_actions[0]] = list(map(lambda x: self.pieces[x].size, next_actions[1]))
        future_q = self.targets.predict_batch(fake_states)
        future_q = [np.amax(future_q[0], axis=1), np.amax(future_q[1], axis=1)]
            
        updated_q = list(map(lambda x: np.add(rewards, (1 - np.array(completes)) * self.GAMMA * x), future_q))
        
        y[0][num_actions[0], actions[0]] = updated_q[0]
        y[1][num_actions[1], actions[1]] = updated_q[1]
        
        if self.use_PER:
            abs_error = np.abs(q_s_a[0][num_actions[0], actions[0]] - updated_q[0]) + np.abs(q_s_a[1][num_actions[1], actions[1]] - updated_q[1])
            self.memory.update(tree_idxs, abs_error)
         
        data = self.model.train_batch(x, {'location':y[0], 'piece':y[1]}, self.BATCH_SIZE)
        
        self.loc_accuracy.append(data.history['location_accuracy'][0])
        self.piece_accuracy.append(data.history['piece_accuracy'][0])
        self.loc_loss.append(data.history['location_loss'][0])
        self.piece_loss.append(data.history['piece_loss'][0])
        self.decay_exploration_rate()

    def decay_exploration_rate(self):
        self.exploration_rate = self.EPSILON_MIN + (self.EPSILON_MAX - self.EPSILON_MIN) * math.exp(-1*self.LAMBDA * self.epsilon_decay_steps)
        self.epsilon_decay_steps += 1
    
    def create_pieces(self):
        self.pieces = np.array([[Piece(j, 4-j, j, i*self.env.NUM_COLS + j) for j in range(4)] for i in range(3)]).reshape(-1)
        
    def reset(self, reward):
        # samples should be of the form (state, action, reward, next_state, complete)
        while len(self.samples) > 0:
            sample = self.samples[0]
            sample.insert(2, reward)
            if not sample[4]:
                try:
                    sample[3] = self.samples[1][0]
                except IndexError:
                    # if the game wasn't over when the player played, but ended
                    # the next move, have None as the next state
                    sample[3] = None
                    
            else:
                sample[3] = None
            
            self.memory.add_sample(tuple(sample))
            self.samples.pop(0)
            if reward < 0:
                # the agent made an illegal move here
                # this reard shoouldn't affect other states in the game,
                # so we exit the loop
                self.samples = []
                break
                
        self.total_rewards.append(self.total_rewards[-1] + reward)
        self.average_reward.append(self.total_rewards[-1]/len(self.total_rewards))
        self.regret.append(len(self.total_rewards) - self.total_rewards[-1])

        self.create_pieces()
        self.pieces_on_board = []
        
    def save_policy(self, prefix):
        fout = open("{}policy_{}".format(prefix, self.name), 'wb')
        pickle.dump(self.model, fout)
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
    
    @property
    def total_pieces(self):
        return np.concatenate(np.array(self.pieces).reshape(-1), self.pieces_on_board)
    
    def update_targets(self):
        self.model.copy_weights(self.targets)
    
    def get_metrics(self):
        return {'loc_loss': self.loc_loss,
                'piece_loss': self.piece_loss,
                'loc_accuracy': self.loc_accuracy,
                'piece_accuracy': self.piece_accuracy,
                'reward': self.total_rewards,
                'average_reward': self.average_reward,
                'regret': self.regret,
                'invalid_moves': self.invalid_moves}
    
    def __str__(self):
        return "{}: {}".format(self.name, self.pieces)