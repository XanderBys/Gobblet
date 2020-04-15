import copy
class State:
    def __init__(self, board):
        self.board = board
        self.lower_layers = [[[0 for i in range(len(board))] for j in range(len(board))] for k in range(len(board)-1)]
        self.hash = str(board)

    def __eq__(self, other):
        transformations = self.get_transformations()
        lower_check = True
        for idx, i in enumerate(self.lower_layers):
            trans = self.get_transformations(i)
            if trans != other.lower_layers[idx]:
                lower_check = False
                break
        
        return other.board in transformations and lower_check
    
    def __hash__(self):
        return hash((str(self.board), str(self.lower_layers)))
    
    def get_transformations(self, default_board=True):
        board = None
        if default_board:
            board = self.board
        else:
            board = default_board
            
        cols = self.get_cols(board)
        
        # 1st rotation (pi/2 radians counter-clockwise)
        r1 = [col[::-1] for col in cols]
        # 2nd rotation (pi radians)
        r2 = [col[::-1] for col in self.get_cols(r1)]
        # 3rd rotation (3pi/2 radians)
        r3 = [col[::-1] for col in self.get_cols(r2)]
        
        transformations = [r1, r2, r3, board]
    
        x_reflection = board[::-1]
        y_reflection = [i[::-1] for i in board]
        transformations.extend([x_reflection, y_reflection])
    
        return transformations
    
    def get_cols(self, arr):
        return [[arr[j][i] for j in range(len(arr[i]))] for i in range(len(arr))]
         
    def __str__(self):
        return str(hash(self))