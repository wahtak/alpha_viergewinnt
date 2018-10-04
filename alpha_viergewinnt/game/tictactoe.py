import numpy as np

from .board import Board, Player, RowOrder
from .condition import ConditionChecker, NStonessInRowCondition
from .alternating_player import AlternatingPlayer


class IllegalMoveException(Exception):
    pass


class FieldOccupiedException(Exception):
    pass


class FreeplayBoard(object):
    '''Functionality for placing stones.'''

    def play_move(self, player, move):
        try:
            field = self.state[move]
        except IndexError:
            raise IllegalMoveException('field %s does not exist.' % (move, ))

        if field != 0:
            raise FieldOccupiedException('field %s is occupied' % (move, ))

        self.state[move] = player.value

    def get_all_moves(self):
        return list(np.ndindex(self.state.shape))

    def get_possible_moves(self):
        return [tuple(index) for index in np.argwhere(self.state == 0)]


class WinCondition(NStonessInRowCondition):
    '''Winning condition for the game Tictactoe.'''

    def __init__(self, player):
        super().__init__(num_stones_in_row=3, player=player)


class DrawCondition(object):
    def check(self, board):
        return len(board.get_possible_moves()) == 0


class Game(Board, FreeplayBoard, AlternatingPlayer, ConditionChecker):
    '''Combination of board, condition checker and alternating player with parameters of the game Tictactoe.'''

    def __init__(self):
        self.board_size = (3, 3)
        Board.__init__(self, size=self.board_size, output_row_order=RowOrder.NORMAL)
        AlternatingPlayer.__init__(self, starting_player=Player.X)

    def __hash__(self):
        return Board.__hash__(self)

    def play_move(self, player, move):
        AlternatingPlayer.register_player_turn(self, player)
        FreeplayBoard.play_move(self, player, move)
