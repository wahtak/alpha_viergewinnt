import pytest

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.game.tictactoe import Game, WinCondition, DrawCondition
from alpha_viergewinnt.player.alpha_player import AlphaPlayer, SelectionStrategy, ConditionEvaluationModel


@pytest.fixture()
def game():
    return Game()


@pytest.fixture()
def alpha_player():
    exploration_factor = 1
    selection_stategy = SelectionStrategy(exploration_factor)
    win_condition = WinCondition(Player.X)
    loss_condition = WinCondition(Player.O)
    draw_condition = DrawCondition()
    evaluation_model = ConditionEvaluationModel(win_condition, loss_condition, draw_condition)
    alpha_player = AlphaPlayer(selection_stategy, evaluation_model)
    return alpha_player


def test_get_any_next_move(game, alpha_player):
    selected_move = alpha_player.get_next_move(game)
    assert selected_move in game.get_possible_moves()
