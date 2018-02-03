from random import Random


class RandomPlayer(object):
    def __init__(self, random=Random()):
        self.random = random

    def get_next_move(self, state):
        possible_moves = state.get_possible_moves()
        selected_move = self.random.choice(possible_moves)
        return selected_move
