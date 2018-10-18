import pytest
import random
from copy import deepcopy

from alpha_viergewinnt.game.board import *
from alpha_viergewinnt.game.viergewinnt import *
from alpha_viergewinnt.game.condition import *
from alpha_viergewinnt.game.alternating_player import *


@pytest.fixture
def game():
    return Viergewinnt()


def test_get_all_moves(game):
    game.play_move(player=Player.X, move=1)
    game.play_move(player=Player.O, move=1)
    game.play_move(player=Player.X, move=1)
    game.play_move(player=Player.O, move=1)
    game.play_move(player=Player.X, move=1)
    game.play_move(player=Player.O, move=1)
    expected_all_moves = [0, 1, 2, 3, 4, 5, 6]
    all_moves = game.get_all_moves()
    assert set(expected_all_moves) == set(all_moves)


def test_get_possible_moves(game):
    game.play_move(player=Player.X, move=1)
    game.play_move(player=Player.O, move=1)
    game.play_move(player=Player.X, move=1)
    game.play_move(player=Player.O, move=1)
    game.play_move(player=Player.X, move=1)
    game.play_move(player=Player.O, move=1)
    expected_possible_moves = [0, 2, 3, 4, 5, 6]
    assert set(expected_possible_moves) == set(game.get_possible_moves())


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


def test_win(game):
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
    assert not game.is_winner(Player.X)
    assert not game.is_winner(Player.O)

    game.play_move(player=Player.X, move=5)
    print(game)
    assert game.is_winner(Player.X)
    assert not game.is_winner(Player.O)

    game.play_move(player=Player.O, move=6)
    print(game)
    assert game.is_winner(Player.X) is True
    assert game.is_winner(Player.O) is True


def test_alternating_turn(game):
    game.play_move(player=Player.X, move=0)
    with pytest.raises(NotPlayersTurnException):
        game.play_move(player=Player.X, move=1)


def test_random_playout_until_full(game):
    max_number_of_moves = 6 * 7
    for move in range(max_number_of_moves):
        assert not game.is_draw()
        random_move = random.choice(game.get_possible_moves())
        game.play_move(player=game.active_player, move=random_move)

    assert game.is_draw()


def test_move_history_inequality(game):
    game1 = deepcopy(game)
    game2 = deepcopy(game)

    game1.play_move(player=Player.X, move=0)
    game2.play_move(player=Player.X, move=0)

    assert game1 == game2

    game1.play_move(player=Player.O, move=1)
    game1.play_move(player=Player.X, move=2)
    game1.play_move(player=Player.O, move=3)

    game2.play_move(player=Player.O, move=3)
    game2.play_move(player=Player.X, move=2)
    game2.play_move(player=Player.O, move=1)

    from alpha_viergewinnt.game.move_recorder import MoveRecorder
    print(MoveRecorder.__hash__(game1))
    print(MoveRecorder.__hash__(game2))
    assert game1 != game2
