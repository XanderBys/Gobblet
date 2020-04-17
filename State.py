import copy
import numpy as np

class State:
    hashes = {}
    
    def __init__(self, board):
        self.board = board
        # dimensions of lower_layers = NUM_ROWS x NUM_COLS x DEPTH-1
        self.lower_layers = np.array([[[0 for i in range(len(board))] for j in range(len(board))] for k in range(len(board)-1)])
        self.hash = str(board)

    def __eq__(self, other):
        get_transformations = self.get_transformations
        arr_eq = np.array_equal
        transformations = get_transformations()
        lower_check = True
        for idx, i in enumerate(self.lower_layers):
            trans = get_transformations(i)
            if arr_eq(trans, other.lower_layers[idx]):
                lower_check = False
                break

        return np.isin(other.board, transformations)[0].all() and lower_check
    
    def __hash__(self):
        return hash((str(self.board), str(self.lower_layers)))
    
    def get_transformations(self, default_board=True):
        board = None
        try:
            val = len(default_board)
            board = default_board
        except TypeError:
            board = self.board
        
        # 1st rotation (pi/2 radians counter-clockwise)
        r1 = board.copy()
        np.rot90(r1)
        # 2nd rotation (pi radians)
        r2 = board.copy()
        np.rot90(r2, 2)
        # 3rd rotation (3pi/2 radians)
        r3 = board.copy()
        np.rot90(r2, 3)
        
        transformations = [r1, r2, r3, board]
    
        x_reflection = board.copy()
        x_reflection = np.flip(x_reflection, 0)
        y_reflection = board.copy()
        y_reflection = np.flip(y_reflection, 1)
        transformations.extend([x_reflection, y_reflection])
    
        return transformations
    
    def get_empty_lower_layer(self):
        return State(self.board.copy())
    
    def __str__(self):
        return str(hash(self))