from .board import Board, Player, RowOrder
from .condition import ConditionChecker, NStonessInRowCondition, NoMovesPossibleCondition
from .alternating_player import AlternatingPlayer
from .move_recorder import MoveRecorder


class IllegalMoveException(Exception):
    pass


class ColumnFullException(Exception):
    pass


class DropdownBoard(object):
    """Functionality for dropping stones."""

    def play_move(self, player, move):
        try:
            column_vector = self.state.T[move]
        except IndexError:
            raise IllegalMoveException("column %d does not exist." % move)

        for index, element in enumerate(column_vector):
            if element == 0:
                column_vector[index] = player.value
                break
        else:
            raise ColumnFullException("column %d is full" % move)

    def get_all_moves(self):
        return list(range(self.state.shape[0] + 1))

    def get_possible_moves(self):
        return [column for column, column_vector in enumerate(self.state.T) if (column_vector == 0).any()]


class Viergewinnt(Board, DropdownBoard, AlternatingPlayer, ConditionChecker, MoveRecorder):
    """
    Combination of board, condition checker, alternating player and move recorder
    with parameters of the game Viergewinnt.
    """

    def __init__(self):
        self.board_size = (6, 7)
        Board.__init__(self, size=self.board_size, output_row_order=RowOrder.REVERSED)
        AlternatingPlayer.__init__(self, starting_player=Player.X)
        MoveRecorder.__init__(self)
        ConditionChecker.__init__(
            self,
            win_conditions={
                Player.X: NStonessInRowCondition(num_stones_in_row=4, player=Player.X),
                Player.O: NStonessInRowCondition(num_stones_in_row=4, player=Player.O),
            },
            draw_condition=NoMovesPossibleCondition(),
        )

    def __hash__(self):
        return Board.__hash__(self) ^ MoveRecorder.__hash__(self)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def play_move(self, player, move):
        AlternatingPlayer.register_player_turn(self, player)
        DropdownBoard.play_move(self, player, move)
        MoveRecorder.record_move(self, move)
