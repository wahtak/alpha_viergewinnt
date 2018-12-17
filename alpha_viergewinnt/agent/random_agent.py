from random import Random


class RandomAgent(object):
    def __init__(self, random_seed=None, **kwargs):
        self.random = Random(random_seed)

    def get_next_move(self, state):
        possible_moves = state.get_possible_moves()
        selected_move = self.random.choice(possible_moves)
        return selected_move
