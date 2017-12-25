import enum

import numpy as np

from .board import Board


class IllegalMoveException(Exception):
    pass


class ColumnFullException(Exception):
    pass


class DropdownBoard(Board):
    '''Board with dropping stones.'''

    def __init__(self, **kwargs):
        super(DropdownBoard, self).__init__(**kwargs)

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


class NStonessInRowWinningCondition(Board):
    '''Winning condition checker for n stones in a row'''

    def __init__(self, num_stones_in_row, **kwargs):
        super(NStonessInRowWinningCondition, self).__init__(**kwargs)

        horizontal_layout = np.ones(num_stones_in_row)
        vertical_layout = np.ones(num_stones_in_row).T
        diagonal_positive_layout = np.diag(np.ones(num_stones_in_row))
        diagonal_negative_layout = np.flip(np.diag(np.ones(num_stones_in_row)), axis=0)
        self.winning_layouts = [horizontal_layout, vertical_layout, diagonal_positive_layout, diagonal_negative_layout]

    def check_win(self, player):
        for layout in self.winning_layouts:
            # todo 2d convolution with state
            pass
        return False


class ViergewinntGame(DropdownBoard, NStonessInRowWinningCondition):
    '''Combination of board and winning condition with parameters of the game Viergewinnt'''

    def __init__(self):
        super(ViergewinntGame, self).__init__(num_stones_in_row=4, size=(6, 7))