import pytest

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.game.tictactoe import Tictactoe
from alpha_viergewinnt.agent.alpha.factory import create_generic_estimator, create_alpha_agent, create_alpha_trainer


@pytest.fixture
def game():
    return Tictactoe()


@pytest.fixture
def alpha_agent(game):
    estimator = create_generic_estimator(game)
    return create_alpha_agent(estimator=estimator, player=Player.X, mcts_steps=30, random_seed=0)


@pytest.fixture
def alpha_trainer(game):
    estimator = create_generic_estimator(game)
    return create_alpha_trainer(estimator=estimator, player=Player.X, mcts_steps=30, random_seed=0)


def test_get_any_next_move(game, alpha_agent):
    selected_move = alpha_agent.get_next_move(game)
    assert selected_move in game.get_possible_moves()


def test_select_winning_move(game, alpha_agent):
    game.play_move(player=Player.X, move=0)
    game.play_move(player=Player.O, move=3)
    game.play_move(player=Player.X, move=4)
    game.play_move(player=Player.O, move=1)
    print(game)

    selected_move = alpha_agent.get_next_move(game)
    assert selected_move == 8


def test_select_non_losing_move(game, alpha_agent):
    game.play_move(player=Player.X, move=3)
    game.play_move(player=Player.O, move=0)
    game.play_move(player=Player.X, move=1)
    game.play_move(player=Player.O, move=4)
    print(game)

    selected_move = alpha_agent.get_next_move(game)
    assert selected_move == 8


def test_train_after_finished_game(game, alpha_trainer):
    game.play_move(player=Player.X, move=0)
    game.play_move(player=Player.O, move=3)
    game.play_move(player=Player.X, move=4)
    game.play_move(player=Player.O, move=1)
    print(game)

    selected_move = alpha_trainer.get_next_move(game)
    assert selected_move == 8

    game.play_move(player=Player.X, move=selected_move)
    alpha_trainer.train(final_state=game)
