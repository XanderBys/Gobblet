import copy
from State import State
from Player import Player

class Environment:
    def __init__(self, NUM_ROWS, NUM_COLS, DEPTH):
        self.state = None
        self.turn = None
        self.NUM_ROWS = NUM_ROWS
        self.NUM_COLS = NUM_COLS
        self.DEPTH = DEPTH
        
        self.reset()
        
    def reset(self):
        # resets the board to be empty and the turn to be 'X'
        self.state = State([[0 for i in range(self.NUM_COLS)] for j in range(self.NUM_COLS)])
        self.turn = 1
    
    def update(self, action, player):
        # updates the board given an action represented as 2 indicies e.g. [0, 2]
        # returns [next_state, result]
        # where next_state is the board after action is taken
        if action not in self.get_legal_moves(player):
            raise ValueError("The action {} is not legal".format(action))
        
        # update the board and the player
        prev_occupant = int(self.state.board[action['destination'][0]][action['destination'][1]])
        self.state.board[action['destination'][0]][action['destination'][1]] = self.turn * action['size']

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
            self.state.board[action['origin'][0]][action['origin'][1]] = 0
        
        if prev_occupant != 0:
            self.update_lower_layers(action, player, prev_occupant)
            
        # update the turn tracker
        self.turn *= -1
        
        return [self.state, self.get_result(self.state)]
    
    def update_lower_layers(self, action, player, prev_occupant, i=0):
        layer = self.state.lower_layers[i]
        dest = layer[action['destination'][0]][action['destination'][1]]
        if dest != 0:
            self.update_lower_layers(self, action, player, dest, i+1)
        dest = self.turn * action['size']
        self.state.lower_layers[i+1][action['destination'][0]][action['destination'][1]] = prev_occupant
        
    def get_result(self, state):
        # returns None if the game isn't over, 1 if white wins and -1 if black wins
        
        # check rows
        for row in state.board:
            ones = list(map(lambda x:(0 if x==0 else (1 if x>0 else -1)), row))
            if abs(sum(ones)) == self.NUM_ROWS:
                return sum(ones) / self.NUM_ROWS
            
        # check columns
        for col in state.get_cols(state.board):
            ones = list(map(lambda x:(0 if x==0 else (1 if x>0 else -1)), col))
            if abs(sum(ones)) == self.NUM_COLS:
                return sum(ones) / self.NUM_COLS
        
        # check diagonals
        diags = [[state.board[i][i] for i in range(len(state.board))], [state.board[(self.NUM_COLS-1)-i][i] for i in range(len(state.board))]]
        for diag in diags:
            ones = list(map(lambda x:(0 if x==0 else (1 if x>0 else -1)), diag))
            if abs(sum(ones)) == self.NUM_ROWS:
                return sum(ones) / self.NUM_ROWS
        
        return None
    
    def get_legal_moves(self, player):
        # returns the legal moves that can be taken
        moves = []
        
        for idx, i in enumerate(self.state.board):
            for jIdx, j in enumerate(i):
                for stack in player.pieces:
                    if len(stack) == 0:
                        continue
                    if self.is_valid_move((idx, jIdx), stack[0]['size']):
                        moves.append({'destination':(idx, jIdx), 'size':int(stack[0]['size']), 'origin':[0]})
                
                for piece in player.pieces_on_board:
                    if self.is_valid_move((idx, jIdx), piece['size']):
                        moves.append({'destination':(idx, jIdx), 'size':int(piece['size']), 'origin':copy.deepcopy(piece['location'])})
        
        return moves
    
    def is_valid_move(self, location, size):
        destination = self.state.board[location[0]][location[1]]
        try:
            return destination == 0 or abs(destination) < size
        except TypeError:
            print(size)
    
    def display(self):
        for i in self.state.board:
            print(i)
        print()