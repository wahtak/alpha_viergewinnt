import pytest

from .board import *
from .viergewinnt import *


@pytest.fixture
def viergewinnt_game():
    return ViergewinntGame()


@pytest.fixture
def player_x_win_condition():
    return NStonessInRowCondition(num_stones_in_row=4, player=Player.X)


@pytest.fixture
def player_o_win_condition():
    return NStonessInRowCondition(num_stones_in_row=4, player=Player.O)


def test_viergewinnt_play_move(viergewinnt_game):
    # valid move
    viergewinnt_game.play_move(player=Player.X, move=1)
    assert viergewinnt_game.state[0, 1] == Player.X.value

    # column does not exist
    with pytest.raises(IllegalMoveException):
        viergewinnt_game.play_move(player=Player.X, move=7)

    # column full
    for _ in range(6):
        viergewinnt_game.play_move(player=Player.X, move=2)
    with pytest.raises(ColumnFullException):
        viergewinnt_game.play_move(player=Player.X, move=2)


def test_viergewinnt_win(viergewinnt_game, player_x_win_condition):
    viergewinnt_game.play_move(player=Player.X, move=2)
    viergewinnt_game.play_move(player=Player.O, move=3)
    viergewinnt_game.play_move(player=Player.X, move=3)
    viergewinnt_game.play_move(player=Player.O, move=4)
    viergewinnt_game.play_move(player=Player.O, move=4)
    viergewinnt_game.play_move(player=Player.X, move=4)
    viergewinnt_game.play_move(player=Player.O, move=5)
    viergewinnt_game.play_move(player=Player.O, move=5)
    viergewinnt_game.play_move(player=Player.O, move=5)
    print(viergewinnt_game)
    assert viergewinnt_game.check(player_x_win_condition) == False

    viergewinnt_game.play_move(player=Player.X, move=5)
    print(viergewinnt_game)
    assert viergewinnt_game.check(player_x_win_condition) == True