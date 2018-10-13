import pytest

from alpha_viergewinnt.player.alpha_player import *

from .test_mcts import DummyState
from .test_mcts import max_first_model as max_first_model_


@pytest.fixture
def max_first_model():
    return max_first_model_()


def test_get_next_move(max_first_model):
    player = AlphaPlayer(max_first_model, mcts_steps=10)
    root = DummyState()
    action = player.get_next_move(root)

    assert action == root.get_possible_moves()[0]
