import numpy as np

from .board import Board, Player, RowOrder
from .condition import ConditionChecker, NStonessInRowCondition, NoMovesPossibleCondition
from .alternating_player import AlternatingPlayer
from .move_recorder import MoveRecorder


class IllegalMoveException(Exception):
    pass


class FieldOccupiedException(Exception):
    pass


class FreeplayBoard(object):
    '''Functionality for placing stones.'''

    def play_move(self, player, move):
        state_index = (move // self.state.shape[1], move % self.state.shape[1])
        try:
            field = self.state[state_index]
        except IndexError:
            raise IllegalMoveException('field %s does not exist.' % (move, ))

        if field != 0:
            raise FieldOccupiedException('field %s is occupied' % (move, ))

        self.state[state_index] = player.value

    def get_all_moves(self):
        return list(range(self.state.size))

    def get_possible_moves(self):
        return np.flatnonzero(self.state.reshape(-1) == 0).tolist()


class Tictactoe(Board, FreeplayBoard, AlternatingPlayer, ConditionChecker, MoveRecorder):
    '''
    Combination of board, condition checker, alternating player and move recorder
    with parameters of the game Tictactoe.
    '''

    def __init__(self):
        self.board_size = (3, 3)
        Board.__init__(self, size=self.board_size, output_row_order=RowOrder.NORMAL)
        AlternatingPlayer.__init__(self, starting_player=Player.X)
        MoveRecorder.__init__(self)
        ConditionChecker.__init__(
            self,
            win_conditions={
                Player.X: NStonessInRowCondition(num_stones_in_row=3, player=Player.X),
                Player.O: NStonessInRowCondition(num_stones_in_row=3, player=Player.O)
            },
            draw_condition=NoMovesPossibleCondition())

    def __hash__(self):
        return Board.__hash__(self) ^ MoveRecorder.__hash__(self)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def play_move(self, player, move):
        AlternatingPlayer.register_player_turn(self, player)
        FreeplayBoard.play_move(self, player, move)
        MoveRecorder.record_move(self, move)
