import pytest
import random

from alpha_viergewinnt.game.board import *
from alpha_viergewinnt.game.tictactoe import *
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
    game.play_move(player=Player.X, move=(0, 1))
    assert game.state[0, 1] == Player.X.value

    # field does not exist
    with pytest.raises(IllegalMoveException):
        game.play_move(player=Player.O, move=(0, 3))


def test_get_all_moves(game):
    game.play_move(player=Player.X, move=(0, 1))
    expected_all_moves = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    all_moves = game.get_all_moves()
    assert set(expected_all_moves) == set(all_moves)


def test_get_possible_moves(game):
    game.play_move(player=Player.X, move=(0, 1))
    expected_possible_moves = [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    assert set(expected_possible_moves) == set(game.get_possible_moves())


def test_win(game, player_x_win_condition, player_o_win_condition):
    game.play_move(player=Player.X, move=(0, 0))
    game.play_move(player=Player.O, move=(0, 1))
    game.play_move(player=Player.X, move=(1, 1))
    game.play_move(player=Player.O, move=(0, 2))

    print(game)
    assert game.check(player_x_win_condition) is False
    assert game.check(player_o_win_condition) is False

    game.play_move(player=Player.X, move=(2, 2))
    print(game)
    assert game.check(player_x_win_condition) is True
    assert game.check(player_o_win_condition) is False


def test_alternating_turn(game):
    game.play_move(player=Player.X, move=(0, 0))
    with pytest.raises(NotPlayersTurnException):
        game.play_move(player=Player.X, move=(0, 1))


def test_random_playout_until_full(game):
    draw_condition = DrawCondition()
    max_number_of_moves = 9
    for move in range(max_number_of_moves):
        assert game.check(draw_condition) is False
        random_move = random.choice(game.get_possible_moves())
        game.play_move(player=game.active_player, move=random_move)

    assert game.check(draw_condition) is True
