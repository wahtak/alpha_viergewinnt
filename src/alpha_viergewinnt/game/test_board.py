import pytest
from copy import deepcopy

from .board import *


@pytest.fixture
def board():
    return Board(size=(6, 7), output_row_order=RowOrder.NORMAL)


def test_board_output(board):
    print(board)


def test_board_set_get(board):
    board.state[0, 1] = 1
    assert board.state[0, 1] == 1
    print(board)


def test_board_hash(board):
    initial_hash = hash(board)
    board.state[0, 1] = 1
    hash_after_update = hash(board)
    assert initial_hash != hash_after_update


def test_board_copyable_and_equality(board):
    board1 = board
    board2 = deepcopy(board)
    assert board1 == board2

    board1.state[0, 1] = 1
    board2.state[0, 2] = 2
    assert board1 != board2

    board1.state[0, 2] = 2
    board2.state[0, 1] = 1
    assert board1 == board2
