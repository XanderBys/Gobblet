import copy
import numpy as np
from State import State
from Player import Player

class Environment:
    def __init__(self, NUM_ROWS, NUM_COLS, DEPTH, init_board=None):
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
    
    def update(self, action, player, turn=0, check_legal=False):
        # updates the board given an action represented as 2 indicies e.g. [0, 2]
        # returns [next_state, result]
        # where next_state is the board after action is taken
        piece, location = action
        if not self.is_legal(action, player):
            if check_legal:
                print(self.state)
                raise ValueError("The action {} is not legal".format(action))
            else:
                return (self.state, 10*self.turn)
        
        if turn == 0:
            turn = self.turn
        
        if piece.is_on_board:
            # if the piece was on the board, set its origin to be empty
            self.state.board[location] = 0
            
        # update the board and the player
        prev_state = copy.copy(self.state)
        action_made = {"prev_state": prev_state}
        
        prev_occupant = int(self.state.board[location])
        self.state.board[location] = turn * piece.size
        
        final_state = self.state
        action_made.update({"final_state": final_state})
        if str(action_made) in self.duplicate_moves:
            self.draw_flag = True
        elif str(action_made) in self.moves_made:
            self.duplicate_moves.add(str(action_made))
        else:
            self.moves_made.add(str(action_made))
        
        for idx, i in enumerate(player.pieces):
            condition = None
            try:
                condition = i.size == piece.size and not i.is_on_board and not piece.is_on_board
            except IndexError:
                continue
            if condition:
                # update values for the locations of pieces
                for idx, i in enumerate(player.pieces[piece.stack_number*4+piece.location:]):
                    if i.is_top_of_stack:
                        break
                    player.pieces[idx].location -= 1
                    
                break
        
        if self.state.lower_layers[0][location] != 0:
            self.state.board[location] = self.state.lower_layers[0][location]
            
        if prev_occupant != 0:
            self.update_lower_layers(action, player, prev_occupant)
            
        # update the turn tracker
        self.turn *= -1
        
        return (self.state, self.get_result(self.state))
    
    def update_lower_layers(self, action, player, prev_occupant, i=0):
        piece, location = action
        layer = self.state.lower_layers[i]
        dest = layer[location]
        if dest != 0:
            self.update_lower_layers(self, action, player, dest, i+1)
        dest = self.turn * piece.size
        self.state.lower_layers[i+1, location[0], location[1]] = prev_occupant
        for p in player.pieces:
            if p.location == location:
                p.stack_number += 1
                break
        
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
    
    def is_legal(self, action, player):
        piece, location = action
        curr_piece = self.state.board[location]
        
        # the piece has to be bigger than the one currently there
        if piece.size <= curr_piece:
            return False
        
        # implement the rule that a new gobblet on the board must be on an empty space
        if not piece.is_on_board and curr_piece != 0:
            # exception: if there is three in a row through the desired location, the move is valid
            row = self.state.board[location[0]]
            col = self.state.board[:, location[1]]
            diag = [0 for i in range(self.NUM_ROWS)]
            if location[0]==location[1]:
                diag = self.state.board.diagonal()
            elif location[0]+location[1] == self.NUM_ROWS-1:
                diag = np.fliplr(self.state.board).diagonal()
            
            flag = False
            
            for i in [row, col, diag]:
                if flag:
                    break
                counter = 0
                for j in np.squeeze(i):
                    if j != 0:
                        counter += 1
                    if counter==3:
                        flag = True
                        break
            
            if not flag:
                return False
        
        return True    

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
                    if is_valid_move((idx, jIdx), stack[0].size):
                        add_move([(idx, jIdx), int(stack[0].size), [0]])
                
                for piece in player.pieces_on_board:
                    if is_valid_move((idx, jIdx), piece.size):
                        add_move([(idx, jIdx), int(piece.size), copy.deepcopy(piece[0])])
        
        return moves
    
    def display(self):
        for i in self.state.board:
            print(i)
        print()