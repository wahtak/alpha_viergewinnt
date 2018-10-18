import pytest

from alpha_viergewinnt.player.alpha_player import *

from .test_mcts import DummyState
from .test_mcts import max_first_evaluator as max_first_evaluator_


@pytest.fixture
def max_first_evaluator():
    return max_first_evaluator_()


def test_get_next_move(max_first_evaluator):
    player = AlphaPlayer(max_first_evaluator, mcts_steps=10, exploration_factor=0.1, random_seed=0)
    root = DummyState()
    action = player.get_next_move(root)

    assert action == root.get_possible_moves()[0]
