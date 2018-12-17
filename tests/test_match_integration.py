import pytest

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.game.tictactoe import Tictactoe
from alpha_viergewinnt.game.viergewinnt import Viergewinnt
from alpha_viergewinnt.agent.random_agent import RandomAgent
from alpha_viergewinnt.agent.pure_mcts.factory import create_pure_mcts_agent
from alpha_viergewinnt.agent.alpha.factory import create_mlp_estimator, create_alpha_agent
from alpha_viergewinnt.match import CompetitionMatch

GAME_FACTORIES = [Tictactoe, Viergewinnt]
GAME_FACTORIES_IDS = ['tictactoe', 'viergewinnt']


@pytest.fixture(params=GAME_FACTORIES, ids=GAME_FACTORIES_IDS)
def game(request):
    Game = request.param
    return Game()


def create_test_pure_mcts_agent(game):
    return create_pure_mcts_agent(player=Player.X, random_seed=0, mcts_steps=2, mcts_rollouts=2)


def create_test_alpha_agent(game):
    estimator = create_mlp_estimator(game)
    return create_alpha_agent(estimator=estimator, player=Player.X, mcts_steps=2)


TEST_PLAYER_FACTORIES = [create_test_pure_mcts_agent, create_test_alpha_agent]


@pytest.fixture(params=TEST_PLAYER_FACTORIES)
def agents(request, game):
    agent_factory = request.param
    return {Player.X: agent_factory(game), Player.O: RandomAgent(random_seed=0)}


def test_single_match_smoketest(game, agents):
    match = CompetitionMatch(game=game, agents=agents)
    winner = match.play()
    assert isinstance(winner, Player) or winner is None


def test_comparison_smoketest(game, agents):
    iterations = 3
    match = CompetitionMatch(game=game, agents=agents)
    results = match.compare(iterations)
    assert sum(results.values()) == iterations
