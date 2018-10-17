import pytest
from random import Random

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.game import tictactoe, viergewinnt
from alpha_viergewinnt.player.random_player import RandomPlayer
from alpha_viergewinnt.player.pure_mcts_player import PureMctsPlayer, create_random_choice_strategy
from alpha_viergewinnt.player.alpha_player import AlphaPlayer, MlpEstimator, Evaluator
from alpha_viergewinnt.match import ComparisonMatch


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
        iterations=2,
        rollouts=2)


def create_alpha_player(game_and_conditions, _):
    game, player_x_win_condition, player_o_win_condition, draw_condition = game_and_conditions
    estimator = MlpEstimator(board_size=game.board_size, actions=game.get_all_moves())
    evaluator = Evaluator(
        estimator=estimator,
        player=Player.X,
        opponent=Player.O,
        win_condition=player_x_win_condition,
        loss_condition=player_o_win_condition,
        draw_condition=draw_condition)
    return AlphaPlayer(evaluator, mcts_steps=2)


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
    match = ComparisonMatch(game=game, players=players, win_conditions=win_conditions, draw_condition=draw_condition)
    winner = match.play()
    assert isinstance(winner, Player) or winner is None


def test_comparison_smoketest(setup):
    game, players, win_conditions, draw_condition = setup
    iterations = 3
    match = ComparisonMatch(game=game, players=players, win_conditions=win_conditions, draw_condition=draw_condition)
    results = match.compare(iterations)
    assert sum(results.values()) == iterations
