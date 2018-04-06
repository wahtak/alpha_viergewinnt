from .board import Board, Player, RowOrder
from .condition import ConditionChecker, NStonessInRowCondition
from .alternating_player import AlternatingPlayer


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


class WinCondition(NStonessInRowCondition):
    '''Winning condition for the game Viergewinnt.'''

    def __init__(self, player):
        super().__init__(num_stones_in_row=4, player=player)


class DrawCondition(object):
    def check(self, board):
        return len(board.get_possible_moves()) == 0


class Game(Board, DropdownBoard, AlternatingPlayer, ConditionChecker):
    '''Combination of board, condition checker and alternating player with parameters of the game Viergewinnt.'''

    def __init__(self):
        Board.__init__(self, size=(6, 7), output_row_order=RowOrder.REVERSED)
        AlternatingPlayer.__init__(self, starting_player=Player.X)

    def __hash__(self):
        return Board.__hash__(self)

    def play_move(self, player, move):
        AlternatingPlayer.register_player_turn(self, player)
        DropdownBoard.play_move(self, player, move)
