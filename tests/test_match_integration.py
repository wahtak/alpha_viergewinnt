import pytest

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.game import tictactoe, viergewinnt
from alpha_viergewinnt.player.random_player import RandomPlayer
from alpha_viergewinnt.player.pure_mcts_player.factory import create_pure_mcts_player
from alpha_viergewinnt.player.alpha_player.factory import create_mlp_estimator, create_alpha_player
from alpha_viergewinnt.match import CompetitionMatch

GAME_FACTORIES = [tictactoe.Game, viergewinnt.Game]
GAME_FACTORIES_IDS = ['tictactoe', 'viergewinnt']


@pytest.fixture(params=GAME_FACTORIES, ids=GAME_FACTORIES_IDS)
def game(request):
    Game = request.param
    return Game()


def create_test_pure_mcts_player(game):
    return create_pure_mcts_player(player=Player.X, random_seed=0, mcts_steps=2, mcts_rollouts=2)


def create_test_alpha_player(game):
    estimator = create_mlp_estimator(game)
    return create_alpha_player(estimator=estimator, player=Player.X, mcts_steps=2)


TEST_PLAYER_FACTORIES = [create_test_pure_mcts_player, create_test_alpha_player]


@pytest.fixture(params=TEST_PLAYER_FACTORIES)
def players(request, game):
    player_factory = request.param
    return {Player.X: player_factory(game), Player.O: RandomPlayer(random_seed=0)}


def test_single_match_smoketest(game, players):
    match = CompetitionMatch(game=game, players=players)
    winner = match.play()
    assert isinstance(winner, Player) or winner is None


def test_comparison_smoketest(game, players):
    iterations = 3
    match = CompetitionMatch(game=game, players=players)
    results = match.compare(iterations)
    assert sum(results.values()) == iterations
