import enum
from copy import deepcopy

import numpy as np

from .board import Board, Player


class IllegalMoveException(Exception):
    pass


class ColumnFullException(Exception):
    pass


class DropdownBoard(object):
    '''Functionality for dropping stones.'''

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


class ConditionChecker(object):
    '''Functionality for checking state conditions.'''

    def check(self, condition):
        return condition.check(self.state)


class NStonessInRowCondition(object):
    '''Condition checker for n stones in a row'''

    def __init__(self, num_stones_in_row, player):
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


class NotPlayersTurnException(Exception):
    pass


class AlternatingPlayer(object):
    '''Functionality for checking alternating player turns'''

    def __init__(self, starting_player):
        self.current_player = starting_player

    def register_player_turn(self, player):
        if player != self.current_player:
            raise NotPlayersTurnException('not player %s\'s turn' % player)
        self.current_player = Player.O if self.current_player == Player.X else Player.X


class ViergewinntGame(Board, DropdownBoard, AlternatingPlayer, ConditionChecker):
    '''Combination of board, winning condition checker and alternating player with parameters of the game Viergewinnt.'''

    def __init__(self):
        Board.__init__(self, size=(6, 7))
        AlternatingPlayer.__init__(self, starting_player=Player.X)

    def __hash__(self):
        return Board.__hash__(self)

    def play_move(self, player, move):
        AlternatingPlayer.register_player_turn(self, player)
        DropdownBoard.play_move(self, player, move)
