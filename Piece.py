class Piece:
    def __init__(self, location, size, stack_number, idx):
        self.location = location
        self.size = size
        self.stack_number = stack_number
        self.idx = idx
    
    @property
    def is_on_board(self):
        return isinstance(self.location, tuple)
    
    @property
    def is_top_of_stack(self):
        return not self.is_on_board and self.location == 0
    
    def __str__(self):
        return "{}, {}".format(self.location, self.size)