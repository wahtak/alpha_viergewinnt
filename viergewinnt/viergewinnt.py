import enum
from copy import deepcopy

import numpy as np

from .board import Board


class IllegalMoveException(Exception):
    pass


class ColumnFullException(Exception):
    pass


class DropdownBoard(Board):
    '''Board with dropping stones.'''

    def play_move(self, player, move):
        try:
            column_vector = self.state.T[move]
        except IndexError:
            raise IllegalMoveException('column %d does not exist.' % move)

        for index, element in enumerate(column_vector):
            if element == 0:
                column_vector[index] = player.value
                break
        else:
            raise ColumnFullException('column %d is full' % move)

    def get_possible_moves(self):
        return [column for column, column_vector in enumerate(self.state.T) if (column_vector == 0).any()]


class ConditionChecker(Board):
    '''Board for checking state conditions.'''

    def check(self, condition):
        return condition.check(self.state)


class NStonessInRowCondition(object):
    '''Condition checker for n stones in a row'''

    def __init__(self, num_stones_in_row, player, **kwargs):
        super(NStonessInRowCondition, self).__init__(**kwargs)
        self.player = player

        horizontal_layout = np.ones(num_stones_in_row)
        vertical_layout = np.ones(num_stones_in_row).T
        regular_diagonal_layout = np.diag(np.ones(num_stones_in_row))
        flipped_diagonal_layout = np.flip(np.diag(np.ones(num_stones_in_row)), axis=0)
        self.winning_layouts = [horizontal_layout, vertical_layout, regular_diagonal_layout, flipped_diagonal_layout]

    def check(self, state):
        for layout in self.winning_layouts:
            # todo 2d convolution with state
            pass
        return False


class ViergewinntGame(DropdownBoard, ConditionChecker):
    '''Combination of board and winning condition checkers with parameters of the game Viergewinnt.'''

    def __init__(self):
        super(ViergewinntGame, self).__init__(size=(6, 7))

    def __hash__(self):
        return Board.__hash__(self)