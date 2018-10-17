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


GAME_FACTORIES = [tictactoe.Game, viergewinnt.Game]

GAME_FACTORIES_IDS = ['tictactoe', 'viergewinnt']


@pytest.fixture(params=GAME_FACTORIES, ids=GAME_FACTORIES_IDS)
def game(request):
    Game = request.param
    return Game()


def create_pure_mcts_player(game, random):
    return PureMctsPlayer(
        player=Player.X,
        selection_strategy=create_random_choice_strategy(random),
        expansion_strategy=create_random_choice_strategy(random),
        simulation_strategy=create_random_choice_strategy(random),
        iterations=2,
        rollouts=2)


def create_alpha_player(game, _):
    estimator = MlpEstimator(board_size=game.board_size, actions=game.get_all_moves())
    evaluator = Evaluator(estimator=estimator, player=Player.X)
    return AlphaPlayer(evaluator, mcts_steps=2)


TEST_PLAYER_FACTORIES = [create_pure_mcts_player, create_alpha_player]


@pytest.fixture(params=TEST_PLAYER_FACTORIES)
def players(request, game, random):
    player_factory = request.param
    return {Player.X: player_factory(game, random), Player.O: RandomPlayer(random)}


def test_single_match_smoketest(game, players):
    match = ComparisonMatch(game=game, players=players)
    winner = match.play()
    assert isinstance(winner, Player) or winner is None


def test_comparison_smoketest(game, players):
    iterations = 3
    match = ComparisonMatch(game=game, players=players)
    results = match.compare(iterations)
    assert sum(results.values()) == iterations
