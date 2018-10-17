import numpy as np
from scipy.signal import convolve2d


class ConditionChecker(object):
    '''Functionality for checking board conditions.'''

    def __init__(self, win_conditions, draw_condition):
        self.win_conditions = win_conditions
        self.draw_condition = draw_condition

    def is_winner(self, player):
        return self.win_conditions[player].check(self)

    def is_draw(self):
        return self.draw_condition.check(self)


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
        player_state = board.get_array_view(player=self.player, player_value=1)
        for layout in self.winning_layouts:
            convolution = convolve2d(player_state, layout, mode='valid')
            if (convolution == self.num_stones_in_row).any():
                return True
        return False


class NoMovesPossibleCondition(object):
    def check(self, board):
        return len(board.get_possible_moves()) == 0
