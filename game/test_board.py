import pytest

from .board import *


@pytest.fixture
def board():
    return Board(size=(6, 7), output_row_order=RowOrder.NORMAL)


def test_board_output(board):
    output_string_length = len(board.__str__())
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