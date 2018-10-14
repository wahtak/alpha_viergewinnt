import pytest

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.game.tictactoe import Game, WinCondition, DrawCondition
from alpha_viergewinnt.player.alpha_player import \
    AlphaPlayer, AlphaTrainer, EvaluationModel, GenericEstimator


@pytest.fixture
def game():
    return Game()


@pytest.fixture
def estimator(game):
    return GenericEstimator(board_size=game.board_size, actions=game.get_all_moves())


@pytest.fixture
def evaluation_model(estimator):
    win_condition = WinCondition(Player.X)
    loss_condition = WinCondition(Player.O)
    draw_condition = DrawCondition()
    return EvaluationModel(
        estimator=estimator,
        player=Player.X,
        opponent=Player.O,
        win_condition=win_condition,
        loss_condition=loss_condition,
        draw_condition=draw_condition)


@pytest.fixture
def alpha_player(evaluation_model):
    return AlphaPlayer(evaluation_model, mcts_steps=30, random_seed=0)


@pytest.fixture
def alpha_trainer(evaluation_model):
    return AlphaTrainer(evaluation_model, mcts_steps=30, random_seed=0)


def test_get_any_next_move(game, alpha_player):
    selected_move = alpha_player.get_next_move(game)
    assert selected_move in game.get_possible_moves()


def test_select_winning_move(game, alpha_player):
    game.play_move(player=Player.X, move=0)
    game.play_move(player=Player.O, move=3)
    game.play_move(player=Player.X, move=4)
    game.play_move(player=Player.O, move=1)
    print(game)

    selected_move = alpha_player.get_next_move(game)
    assert selected_move == 8


def test_select_non_losing_move(game, alpha_player):
    game.play_move(player=Player.X, move=3)
    game.play_move(player=Player.O, move=0)
    game.play_move(player=Player.X, move=1)
    game.play_move(player=Player.O, move=4)
    print(game)

    selected_move = alpha_player.get_next_move(game)
    assert selected_move == 8


def test_learn_after_finished_game(game, alpha_trainer):
    game.play_move(player=Player.X, move=0)
    game.play_move(player=Player.O, move=3)
    game.play_move(player=Player.X, move=4)
    game.play_move(player=Player.O, move=1)
    print(game)

    selected_move = alpha_trainer.get_next_move(game)
    assert selected_move == 8

    game.play_move(player=Player.X, move=selected_move)
    alpha_trainer.learn(final_state=game)
