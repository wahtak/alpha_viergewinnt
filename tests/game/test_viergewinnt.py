import pytest
import random

from alpha_viergewinnt.game.board import *
from alpha_viergewinnt.game.viergewinnt import *
from alpha_viergewinnt.game.condition import *
from alpha_viergewinnt.game.alternating_player import *


@pytest.fixture
def game():
    return Game()


@pytest.fixture
def player_x_win_condition():
    return WinCondition(Player.X)


@pytest.fixture
def player_o_win_condition():
    return WinCondition(Player.O)


def test_play_move(game):
    # valid move
    game.play_move(player=Player.X, move=1)
    assert game.state[0, 1] == Player.X.value

    # column does not exist
    with pytest.raises(IllegalMoveException):
        game.play_move(player=Player.O, move=7)

    # column full
    for _ in range(6):
        game.play_move(player=Player.X, move=2)
        game.play_move(player=Player.O, move=3)
    assert 2 not in game.get_possible_moves()
    with pytest.raises(ColumnFullException):
        game.play_move(player=Player.X, move=2)


def test_win(game, player_x_win_condition, player_o_win_condition):
    game.play_move(player=Player.X, move=2)
    game.play_move(player=Player.O, move=3)
    game.play_move(player=Player.X, move=3)
    game.play_move(player=Player.O, move=4)
    game.play_move(player=Player.X, move=0)
    game.play_move(player=Player.O, move=4)
    game.play_move(player=Player.X, move=4)
    game.play_move(player=Player.O, move=5)
    game.play_move(player=Player.X, move=0)
    game.play_move(player=Player.O, move=5)
    game.play_move(player=Player.X, move=0)
    game.play_move(player=Player.O, move=5)
    print(game)
    assert game.check(player_x_win_condition) is False
    assert game.check(player_o_win_condition) is False

    game.play_move(player=Player.X, move=5)
    print(game)
    assert game.check(player_x_win_condition) is True
    assert game.check(player_o_win_condition) is False

    game.play_move(player=Player.O, move=6)
    print(game)
    assert game.check(player_x_win_condition) is True
    assert game.check(player_o_win_condition) is True


def test_alternating_turn(game):
    game.play_move(player=Player.X, move=0)
    with pytest.raises(NotPlayersTurnException):
        game.play_move(player=Player.X, move=1)


def test_random_playout_until_full(game):
    draw_condition = DrawCondition()
    max_number_of_moves = 6 * 7
    for move in range(max_number_of_moves):
        assert game.check(draw_condition) is False
        random_move = random.choice(game.get_possible_moves())
        game.play_move(player=game.current_player, move=random_move)

    assert game.check(draw_condition) is True