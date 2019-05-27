import pytest
import matplotlib

from alpha_viergewinnt.agent.alpha import *

from .test_mcts import DummyState
from .test_mcts import MaxFirstEvaluator


@pytest.fixture
def max_first_evaluator():
    return MaxFirstEvaluator()


def test_get_next_move(max_first_evaluator):
    matplotlib.use('Agg')
    agent = AlphaAgent(max_first_evaluator, mcts_steps=10, exploration_factor=0.1, random_seed=0, draw_graph=True)
    root = DummyState()
    action = agent.get_next_move(root)

    assert action == root.get_possible_moves()[0]