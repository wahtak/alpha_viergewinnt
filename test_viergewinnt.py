import pytest

from .board import *
from .viergewinnt import *


@pytest.fixture
def viergewinnt_board():
    winning_condition = NStonessInRowWinningCondition(num_stones_in_row=4)
    board = DropdownBoard(winning_condition=winning_condition, size=(6, 7))
    return board


def test_viergewinnt_insert(viergewinnt_board):
    # valid insert
    viergewinnt_board.insert(player=Player.X, column=1)
    assert viergewinnt_board.state[0, 1] == Player.X.value

    # column does not exist
    with pytest.raises(IllegalMoveException):
        viergewinnt_board.insert(player=Player.X, column=7)

    # column full
    for _ in range(6):
        viergewinnt_board.insert(player=Player.X, column=2)
    with pytest.raises(ColumnFullException):
        viergewinnt_board.insert(player=Player.X, column=2)


def test_viergewinnt_win(viergewinnt_board):
    viergewinnt_board.state[2, 0] = 1
    viergewinnt_board.state[2, 1] = 1
    viergewinnt_board.state[2, 2] = 1
    viergewinnt_board.state[2, 3] = 1

    # TODO
    assert False