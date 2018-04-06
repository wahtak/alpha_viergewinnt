import pytest
import random

from .board import *
from .viergewinnt import *
from .condition import *
from .alternating_player import *


@pytest.fixture
def viergewinnt():
    return Viergewinnt()


@pytest.fixture
def player_x_win_condition():
    return WinCondition(Player.X)


@pytest.fixture
def player_o_win_condition():
    return WinCondition(Player.O)


def test_viergewinnt_play_move(viergewinnt):
    # valid move
    viergewinnt.play_move(player=Player.X, move=1)
    assert viergewinnt.state[0, 1] == Player.X.value

    # column does not exist
    with pytest.raises(IllegalMoveException):
        viergewinnt.play_move(player=Player.O, move=7)

    # column full
    for _ in range(6):
        viergewinnt.play_move(player=Player.X, move=2)
        viergewinnt.play_move(player=Player.O, move=3)
    assert 2 not in viergewinnt.get_possible_moves()
    with pytest.raises(ColumnFullException):
        viergewinnt.play_move(player=Player.X, move=2)


def test_viergewinnt_win(viergewinnt, player_x_win_condition, player_o_win_condition):
    viergewinnt.play_move(player=Player.X, move=2)
    viergewinnt.play_move(player=Player.O, move=3)
    viergewinnt.play_move(player=Player.X, move=3)
    viergewinnt.play_move(player=Player.O, move=4)
    viergewinnt.play_move(player=Player.X, move=0)
    viergewinnt.play_move(player=Player.O, move=4)
    viergewinnt.play_move(player=Player.X, move=4)
    viergewinnt.play_move(player=Player.O, move=5)
    viergewinnt.play_move(player=Player.X, move=0)
    viergewinnt.play_move(player=Player.O, move=5)
    viergewinnt.play_move(player=Player.X, move=0)
    viergewinnt.play_move(player=Player.O, move=5)
    print(viergewinnt)
    assert viergewinnt.check(player_x_win_condition) is False
    assert viergewinnt.check(player_o_win_condition) is False

    viergewinnt.play_move(player=Player.X, move=5)
    print(viergewinnt)
    assert viergewinnt.check(player_x_win_condition) is True
    assert viergewinnt.check(player_o_win_condition) is False

    viergewinnt.play_move(player=Player.O, move=6)
    print(viergewinnt)
    assert viergewinnt.check(player_x_win_condition) is True
    assert viergewinnt.check(player_o_win_condition) is True


def test_viergewinnt_alternating_turn(viergewinnt):
    viergewinnt.play_move(player=Player.X, move=0)
    with pytest.raises(NotPlayersTurnException):
        viergewinnt.play_move(player=Player.X, move=1)


def test_random_playout_until_full(viergewinnt):
    draw_condition = DrawCondition()
    max_number_of_moves = 6 * 7
    for move in range(max_number_of_moves):
        assert viergewinnt.check(draw_condition) is False
        random_move = random.choice(viergewinnt.get_possible_moves())
        viergewinnt.play_move(player=viergewinnt.current_player, move=random_move)

    assert viergewinnt.check(draw_condition) is True
