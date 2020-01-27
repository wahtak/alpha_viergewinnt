import pytest
import matplotlib

from alpha_viergewinnt.agent.alpha import *

from .test_mcts import DummyState
from .test_mcts import MaxFirstEvaluator


@pytest.fixture
def max_first_evaluator():
    return MaxFirstEvaluator()


@pytest.mark.filterwarnings(
    "ignore:Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure"
)
def test_get_next_move(max_first_evaluator):
    # use backend which does not require a display for CI
    matplotlib.use("Agg")

    agent = AlphaAgent(evaluator=max_first_evaluator, mcts_steps=10, random_seed=0, draw_graph=True)
    root = DummyState()
    action = agent.get_next_move(root)

    assert action == root.get_possible_moves()[0]
