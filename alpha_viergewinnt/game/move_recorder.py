class MoveRecorder(object):
    def __init__(self):
        self.recorded_moves = tuple()

    def record_move(self, move):
        self.recorded_moves += (move, )

    def __hash__(self):
        return hash(self.recorded_moves)
