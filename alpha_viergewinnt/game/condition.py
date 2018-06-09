import numpy as np
from scipy.signal import convolve2d


class ConditionChecker(object):
    '''Functionality for checking board conditions.'''

    def check(self, condition):
        return condition.check(self)


class NStonessInRowCondition(object):
    '''Condition checker for n stones in a row'''

    def __init__(self, num_stones_in_row, player):
        self.player = player
        self.num_stones_in_row = num_stones_in_row
        horizontal_layout = np.ones((num_stones_in_row, 1))
        vertical_layout = np.ones((1, num_stones_in_row))
        regular_diagonal_layout = np.diag(np.ones(num_stones_in_row))
        flipped_diagonal_layout = np.flip(np.diag(np.ones(num_stones_in_row)), axis=0)
        self.winning_layouts = [horizontal_layout, vertical_layout, regular_diagonal_layout, flipped_diagonal_layout]

    def check(self, board):
        player_state = board.get_player_state(self.player)
        for layout in self.winning_layouts:
            convolution = convolve2d(player_state, layout, mode='valid')
            if (convolution == self.num_stones_in_row).any():
                return True
        return False