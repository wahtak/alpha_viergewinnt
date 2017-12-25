import enum

import numpy as np

from .board import Board


class IllegalMoveException(Exception):
    pass


class ColumnFullException(Exception):
    pass


class DropdownBoard(Board):
    '''Board with dropping stones.'''

    def __init__(self, winning_condition, *args, **kwargs):
        super(DropdownBoard, self).__init__(*args, **kwargs)

    def insert(self, player, column):
        try:
            column_vector = self.state.T[column]
        except IndexError:
            raise IllegalMoveException('column %d does not exist.' % column)

        for index, element in enumerate(column_vector):
            if element == 0:
                column_vector[index] = player.value
                break
        else:
            raise ColumnFullException('column %d is full' % column)

    def check_win(self, player):
        self.winning_condition.check_win(self.state, player)


class NStonessInRowWinningCondition(object):
    '''Winning condition checker for n stones in a row'''

    def __init__(self, num_stones_in_row):
        horizontal_layout = np.ones(num_stones_in_row)
        vertical_layout = np.ones(num_stones_in_row).T
        diagonal_positive_layout = np.diag(np.ones(num_stones_in_row))
        diagonal_negative_layout = np.flip(np.diag(np.ones(num_stones_in_row)), axis=0)
        self.winning_layouts = [horizontal_layout, vertical_layout, diagonal_positive_layout, diagonal_negative_layout]

    def check_win(self, state, player):
        for layout in winning_layouts:
            # todo 2d convolution with state
            pass
