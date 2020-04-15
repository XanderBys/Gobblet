class Piece:
    def __init__(self, location, size):
        # if location is an int, it represents position in the stack of pieces
        # else, it represents location on the board
        self.location = location