import copy
import numpy as np

class State:
    hashes = {}
    
    def __init__(self, board, lower_layers=[]):
        self.board = board
        # dimensions of lower_layers = NUM_ROWS x NUM_COLS x DEPTH-1
        if len(lower_layers) > 0:
            self.lower_layers = lower_layers
        else:
            self.lower_layers = np.array([[[0 for i in range(len(board))] for j in range(len(board))] for k in range(len(board)-1)])
        
        self.hash = str(board)

    def __eq__(self, other):
        get_transformations = self.get_transformations
        arr_eq = np.array_equal
        all_transformations = get_transformations()
        transformations = all_transformations[0]
        
        lower_transformations = all_transformations[1]
        
        return np.isin(other.board, transformations).all() and np.isin(other.board.lower_layers, lower_transformations)
    
    def __hash__(self):
        return hash((str(self.board), str(self.lower_layers)))
    
    def get_transformations(self, default_board=True):
        board = None
        other_board = []
        try:
            val = len(default_board)
            board = default_board
        except TypeError:
            board = self.board
            other_board = self.lower_layers
        
        all_transformations = []

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
        all_transformations.append(transformations)
        
        if len(other_board) != 0:
            # this is the lower layers
            lower_transformations = list(map(self.get_transformations, other_board))
            all_transformations.append(lower_transformations)
            
        return all_transformations if len(all_transformations) > 1 else all_transformations[0]
    
    def get_transformations_states(self):
        transformations = self.get_transformations()
        return list(map(self.__init__, transformations[0], np.array(transformations[1])))
    
    def get_empty_lower_layer(self):
        return State(self.board.copy())
    
    def __str__(self):
        return str([self.board, self.lower_layers])
    
    def deepcopy(self):
        st = State(self.board.copy())
        st.lower_layers = self.lower_layers.copy()
        return st

if __name__ == '__main__':
    st1 = State([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    st2 = State([[1, 0, 0], [0, 0, 0], [0, 0, 0]])

    print(st1==st2)