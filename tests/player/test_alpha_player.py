import pytest

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.game.tictactoe import Game, WinCondition, DrawCondition
from alpha_viergewinnt.player.alpha_player import AlphaPlayer, SelectionStrategy, ConditionEvaluationModel


@pytest.fixture()
def game():
    return Game()


@pytest.fixture()
def selection_stategy():
    exploration_factor = 1
    return SelectionStrategy(exploration_factor)


@pytest.fixture()
def evaluation_model():
    win_condition = WinCondition(Player.X)
    loss_condition = WinCondition(Player.O)
    draw_condition = DrawCondition()
    return ConditionEvaluationModel(win_condition, loss_condition, draw_condition)


def test_get_any_next_move(game, selection_stategy, evaluation_model):
    alpha_player = AlphaPlayer(selection_stategy, evaluation_model, mcts_steps=3)
    selected_move = alpha_player.get_next_move(game)
    assert selected_move in game.get_possible_moves()


def test_select_winning_move(game, selection_stategy, evaluation_model):
    game.play_move(player=Player.X, move=(0, 0))
    game.play_move(player=Player.O, move=(1, 0))
    game.play_move(player=Player.X, move=(1, 1))
    game.play_move(player=Player.O, move=(0, 1))
    print(game)

    alpha_player = AlphaPlayer(selection_stategy, evaluation_model, mcts_steps=30)
    selected_move = alpha_player.get_next_move(game)
    assert selected_move == (2, 2)


def test_select_non_losing_move(game, selection_stategy, evaluation_model):
    game.play_move(player=Player.X, move=(1, 0))
    game.play_move(player=Player.O, move=(0, 0))
    game.play_move(player=Player.X, move=(0, 1))
    game.play_move(player=Player.O, move=(1, 1))
    print(game)

    alpha_player = AlphaPlayer(selection_stategy, evaluation_model, mcts_steps=30)
    selected_move = alpha_player.get_next_move(game)
    assert selected_move == (2, 2)
