import copy
import numpy as np
from State import State
from Player import Player

class Environment:
    def __init__(self, NUM_ROWS, NUM_COLS, DEPTH):
        self.state = None
        self.moves_made = set()
        self.duplicate_moves = set()
        self.draw_flag = False
        self.turn = None
        
        self.NUM_ROWS = NUM_ROWS
        self.NUM_COLS = NUM_COLS
        self.DEPTH = DEPTH
        
        self.reset()
        
    def reset(self):
        # resets the board to be empty and the turn to be 'X'
        self.state = State(np.array([[0 for i in range(self.NUM_COLS)] for j in range(self.NUM_COLS)]))
        self.moves_made = set()
        self.duplicate_moves = set()
        self.draw_flag = False
        self.turn = 1
    
    def update(self, action, player):
        # updates the board given an action represented as 2 indicies e.g. [0, 2]
        # returns [next_state, result]
        # where next_state is the board after action is taken
        if action not in self.get_legal_moves(player):
            raise ValueError("The action {} is not legal".format(action))
        
        # update the board and the player
        prev_state = copy.copy(self.state)
        action_made = {"prev_state": prev_state}
        
        prev_occupant = int(self.state.board[action['destination'][0], action['destination'][1]])
        self.state.board[action['destination'][0], action['destination'][1]] = self.turn * action['size']
        
        final_state = self.state
        action_made.update({"final_state": final_state})
        if str(action_made) in self.duplicate_moves:
            self.draw_flag = True
        elif str(action_made) in self.moves_made:
            self.duplicate_moves.add(str(action_made))
        else:
            self.moves_made.add(str(action_made))
        
        for i in player.pieces:
            condition = None
            try:
                condition = i[0]['size'] == action['size']
            except IndexError:
                continue
            if condition:
                i.pop(0)
                break

        player.pieces_on_board.append({'location': action['destination'], 'size': action['size']})
        
        if len(action['origin']) == 2:
            # if the piece was on the board, set its origin to be empty
            self.state.board[action['origin'][0], action['origin'][1]] = 0
        
        if prev_occupant != 0:
            self.update_lower_layers(action, player, prev_occupant)
            
        # update the turn tracker
        self.turn *= -1
        
        return (self.state, self.get_result(self.state))
    
    def update_lower_layers(self, action, player, prev_occupant, i=0):
        layer = self.state.lower_layers[i]
        dest = layer[action['destination'][0], action['destination'][1]]
        if dest != 0:
            self.update_lower_layers(self, action, player, dest, i+1)
        dest = self.turn * action['size']
        self.state.lower_layers[i+1, action['destination'][0], action['destination'][1]] = prev_occupant
        
    def get_result(self, state):
        # returns None if the game isn't over, 1 if white wins and -1 if black wins
        
        # check rows
        for row in state.board:
            ones = np.sign(row)
            if abs(sum(ones)) == self.NUM_ROWS:
                return sum(ones) / self.NUM_ROWS
            
        # check columns
        cols = state.board.copy()
        cols.transpose()
        for col in cols:
            ones = np.sign(row)
            if abs(sum(ones)) == self.NUM_COLS:
                return sum(ones) / self.NUM_COLS
        
        # check diagonals
        diags = [state.board.diagonal(), np.fliplr(state.board).diagonal()]
        for diag in diags:
            ones = np.sign(diag)
            if abs(sum(ones)) == self.NUM_ROWS:
                return sum(ones) / self.NUM_ROWS
        
        # check for draws
        # that is, if three identical moves have been made, it's a draw
        if self.draw_flag:
            return 0
            
        return None
    
    def get_legal_moves(self, player):
        # returns the legal moves that can be taken
        moves = []
        add_move = moves.append
        is_valid_move = self.is_valid_move
        for idx, i in enumerate(self.state.board):
            for jIdx, j in enumerate(i):
                for stack in player.pieces:
                    if len(stack) == 0:
                        continue
                    if is_valid_move((idx, jIdx), stack[0]['size']):
                        add_move({'destination':(idx, jIdx), 'size':int(stack[0]['size']), 'origin':[0]})
                
                for piece in player.pieces_on_board:
                    if is_valid_move((idx, jIdx), piece['size']):
                        add_move({'destination':(idx, jIdx), 'size':int(piece['size']), 'origin':copy.deepcopy(piece['location'])})
        
        return moves
    
    def is_valid_move(self, location, size):
        destination = self.state.board[location[0], location[1]]
        try:
            return destination == 0 or abs(destination) < size
        except TypeError:
            print(size)
    
    def display(self):
        for i in self.state.board:
            print(i)
        print()