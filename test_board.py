import pytest

from .board import *


def test_board_output():
    board = Board(size=(6, 7))
    output_string_length = len(board.__str__())
    print(board)


def test_board_set_get():
    board = Board(size=(3, 3))
    board.state[0, 1] = 1
    assert board.state[0, 1] == 1
    print(board)