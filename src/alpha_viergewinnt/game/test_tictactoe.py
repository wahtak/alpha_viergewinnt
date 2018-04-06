import pytest
import random

from .board import *
from .tictactoe import *
from .condition import *
from .alternating_player import *


@pytest.fixture
def tictactoe():
    return Tictactoe()


@pytest.fixture
def player_x_win_condition():
    return WinCondition(Player.X)


@pytest.fixture
def player_o_win_condition():
    return WinCondition(Player.O)


def test_tictactoe_play_move(tictactoe):
    # valid move
    tictactoe.play_move(player=Player.X, move=(0, 1))
    assert tictactoe.state[0, 1] == Player.X.value

    # field does not exist
    with pytest.raises(IllegalMoveException):
        tictactoe.play_move(player=Player.O, move=(0, 3))


def test_tictactoe_possible_moves(tictactoe):
    tictactoe.play_move(player=Player.X, move=(0, 1))
    expected_possible_moves = [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    assert set(expected_possible_moves) == set(tictactoe.get_possible_moves())


def test_tictactoe_win(tictactoe, player_x_win_condition, player_o_win_condition):
    tictactoe.play_move(player=Player.X, move=(0, 0))
    tictactoe.play_move(player=Player.O, move=(0, 1))
    tictactoe.play_move(player=Player.X, move=(1, 1))
    tictactoe.play_move(player=Player.O, move=(0, 2))

    print(tictactoe)
    assert tictactoe.check(player_x_win_condition) is False
    assert tictactoe.check(player_o_win_condition) is False

    tictactoe.play_move(player=Player.X, move=(2, 2))
    print(tictactoe)
    assert tictactoe.check(player_x_win_condition) is True
    assert tictactoe.check(player_o_win_condition) is False


def test_tictactoe_alternating_turn(tictactoe):
    tictactoe.play_move(player=Player.X, move=(0, 0))
    with pytest.raises(NotPlayersTurnException):
        tictactoe.play_move(player=Player.X, move=(0, 1))


def test_random_playout_until_full(tictactoe):
    draw_condition = DrawCondition()
    max_number_of_moves = 9
    for move in range(max_number_of_moves):
        assert tictactoe.check(draw_condition) is False
        random_move = random.choice(tictactoe.get_possible_moves())
        tictactoe.play_move(player=tictactoe.current_player, move=random_move)

    assert tictactoe.check(draw_condition) is True
