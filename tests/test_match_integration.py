import pytest
from random import Random

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.game import tictactoe, viergewinnt
from alpha_viergewinnt.player.random_player import RandomPlayer
from alpha_viergewinnt.player.pure_mcts_player import PureMctsPlayer, create_random_choice_strategy
from alpha_viergewinnt.player.alpha_player import AlphaPlayer, GenericEstimator, EvaluationModel, SelectionStrategy
from alpha_viergewinnt.match import Match


@pytest.fixture()
def random():
    # initialize seeded random number generator for deterministic test
    return Random(1)


GAME_FACTORIES = [(tictactoe.Game, tictactoe.WinCondition, tictactoe.DrawCondition),
                  (viergewinnt.Game, viergewinnt.WinCondition, viergewinnt.DrawCondition)]

GAME_FACTORIES_IDS = ['tictactoe', 'viergewinnt']


@pytest.fixture(params=GAME_FACTORIES, ids=GAME_FACTORIES_IDS)
def game_and_conditions(request):
    Game, WinCondition, DrawCondition = request.param
    return Game(), WinCondition(Player.X), WinCondition(Player.O), DrawCondition()


def create_pure_mcts_player(game_and_conditions, random):
    _, win_condition, loss_condition, draw_condition = game_and_conditions
    return PureMctsPlayer(
        win_condition=win_condition,
        loss_condition=loss_condition,
        draw_condition=draw_condition,
        selection_strategy=create_random_choice_strategy(random),
        expansion_strategy=create_random_choice_strategy(random),
        simulation_strategy=create_random_choice_strategy(random),
        iterations=3,
        rollouts=3)


def create_alpha_player(game_and_conditions, _):
    game, win_condition, loss_condition, draw_condition = game_and_conditions
    selection_stategy = SelectionStrategy(exploration_factor=1)
    estimator = GenericEstimator(board_size=game.board_size, actions=game.get_all_moves())
    evaluation_model = EvaluationModel(estimator, win_condition, loss_condition, draw_condition)
    return AlphaPlayer(selection_stategy, evaluation_model, mcts_steps=5)


TEST_PLAYER_FACTORIES = [create_pure_mcts_player, create_alpha_player]


@pytest.fixture(params=TEST_PLAYER_FACTORIES)
def setup(request, game_and_conditions, random):
    player_factory = request.param
    game, player_x_win_condition, player_o_win_condition, draw_condition = game_and_conditions
    players = {Player.X: player_factory(game_and_conditions, random), Player.O: RandomPlayer(random)}
    win_conditions = {Player.X: player_x_win_condition, Player.O: player_o_win_condition}
    return game, players, win_conditions, draw_condition


def test_single_match_smoketest(setup):
    game, players, win_conditions, draw_condition = setup
    match = Match(game=game, players=players, win_conditions=win_conditions, draw_condition=draw_condition)
    winner = match.play()
    assert isinstance(winner, Player) or winner is None


def test_evaluation_smoketest(setup):
    game, players, win_conditions, draw_condition = setup
    iterations = 3
    match = Match(game=game, players=players, win_conditions=win_conditions, draw_condition=draw_condition)
    results = match.evaluate(iterations)
    assert sum(results.values()) == iterations
